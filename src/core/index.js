const { UnifiProvider } = require('../providers/unifi');
const { OnvifProvider } = require('../providers/onvif');
const { VideoProvider } = require('../providers/video');
const { DiscoveryService } = require('../services/discovery');
const { RtspParityBridge } = require('../services/rtsp-parity-bridge');
const { TlsCertificate } = require('../domain/tls-certificate');
const { DetectionGate } = require('../services/detection-gate');

const DECISION_REASON_LABELS = {
  invalid_phase: 'Invalid phase',
  unifi_not_connected: 'UniFi offline',
  unknown_class_blocked: 'Unknown class blocked',
  class_not_enabled: 'Class blocked',
  below_min_score: 'Below threshold',
  gate_opened: 'Gate opened',
  gate_closed: 'Gate closed',
  already_active: 'Already active',
  still_active: 'Still active',
  already_inactive: 'Already inactive',
  no_transition: 'No transition'
};

function getDecisionReasonLabel(reasonCode) {
  if (!reasonCode) return null;
  return DECISION_REASON_LABELS[reasonCode] || 'Other';
}

class CameraProxy {
  constructor(config, logger) {
    this.config = config;
    this.logger = logger;
    this.providers = {};
    this.services = {};
    this.resources = {};
    this.isRunning = false;
    this.motionEventId = 0;
    this.detectionGate = new DetectionGate(this.config.detection);
    this.rtspParityEnabled = !!this.config?.detection?.rtspParity?.enabled;
  }

  async start() {
    this.logger.info('Initializing camera proxy');

    const configDir = require('path').join(require('os').homedir(), '.unifi-camera-proxy');
    if (!require('fs').existsSync(configDir)) {
      require('fs').mkdirSync(configDir, { recursive: true });
    }

    this.resources.tls = new TlsCertificate(this.config.tls, this.logger);
    const tlsPaths = this.resources.tls.ensureExists(configDir);

    this.providers.onvif = new OnvifProvider(this.config.onvif, this.logger);
    await this.providers.onvif.connect();

    this.providers.onvif.on('motion:start', (metadata) => this.handleMotionStart(metadata));
    this.providers.onvif.on('motion:stop', (metadata) => this.handleMotionStop(metadata));

    this.services.discovery = new DiscoveryService(this.config.camera, this.logger);
    this.services.discovery.start();

    this.providers.video = new VideoProvider(this.config.video, this.logger);
    await this.providers.video.initialize();
    this.providers.video.setSourceUrl(this.providers.onvif.getStreamUrl());

    this.providers.unifi = new UnifiProvider({
      ...this.config.unifi,
      ...this.config.camera,
      tlsCert: tlsPaths.cert,
      tlsKey: tlsPaths.key
    }, this.logger);

    await this.providers.unifi.connect();
    this.providers.unifi.setVideoProvider(this.providers.video);

    if (this.rtspParityEnabled) {
      const configuredParity = this.config?.detection?.rtspParity || {};
      const rtspUrl = configuredParity.rtspUrl || this.providers.onvif.getStreamUrl();

      if (!rtspUrl) {
        this.logger.warn('RTSP parity enabled but no stream URL available; parity runner disabled');
        this.rtspParityEnabled = false;
      } else {
        try {
          this.services.rtspParity = new RtspParityBridge(
            {
              ...configuredParity,
              rtspUrl
            },
            this.logger
          );

          this.services.rtspParity.start((message) => this.handleRtspParityMessage(message));
        } catch (err) {
          this.logger.error('Failed to start RTSP parity bridge, falling back to ONVIF motion', err);
          this.rtspParityEnabled = false;
        }
      }
    }

    this.isRunning = true;
    this.logger.info('Camera proxy started successfully');
  }

  handleMotionStart(metadata = {}) {
    if (this.rtspParityEnabled) return;
    this.processDetectionSignal({ phase: 'start', ...metadata });
  }

  handleMotionStop(metadata = {}) {
    if (this.rtspParityEnabled) return;
    this.processDetectionSignal({ phase: 'stop', ...metadata });
  }

  handleRtspParityMessage(message = {}) {
    if (!message || typeof message !== 'object') return;

    if (message.functionName === 'ParityFrameSummary') {
      if (this.config?.detection?.logDecisions) {
        this.logger.debug('RTSP parity frame', message.payload || {});
      }
      return;
    }

    if (message.functionName !== 'EventSmartDetect') return;

    const payload = message.payload && typeof message.payload === 'object'
      ? message.payload
      : null;
    if (!payload) return;

    const edgeType = payload.edgeType === 'enter'
      ? 'start'
      : payload.edgeType === 'leave'
        ? 'stop'
        : null;
    if (!edgeType) return;

    const objectTypes = Array.isArray(payload.objectTypes) ? payload.objectTypes : [];
    const className = objectTypes.find((item) => typeof item === 'string' && item.trim());
    if (!className) return;

    const score = typeof payload.confidence === 'number' && Number.isFinite(payload.confidence)
      ? payload.confidence
      : null;

    this.processDetectionSignal({
      phase: edgeType,
      className,
      score
    });
  }

  processDetectionSignal(signal) {
    if (!this.providers.unifi.isConnected()) {
      this.logDetectionDecision(signal, {
        accepted: false,
        reason: 'unifi_not_connected',
        transition: null,
        token: null,
        phase: signal.phase || null,
        minScore: null,
        scoreValue: null,
        activeCountBefore: null,
        activeCountAfter: null
      });
      return;
    }

    const decision = this.detectionGate.evaluate(signal);
    this.logDetectionDecision(signal, decision);

    const transition = decision.transition;
    if (!transition) return;

    if (transition === 'start') {
      this.logger.debug('Motion detected', {
        className: signal.className || null,
        score: signal.score ?? null
      });
      this.providers.unifi.sendMotionEvent(this.motionEventId, 'start');

      if (this.shouldEmitSmartDetectEvent(decision)) {
        this.providers.unifi.sendSmartDetectEvent(this.motionEventId, 'start', {
          className: decision.token,
          score: decision.scoreValue
        });
      }
      return;
    }

    this.logger.debug('Motion cleared', {
      className: signal.className || null,
      score: signal.score ?? null
    });

    if (this.shouldEmitSmartDetectEvent(decision)) {
      this.providers.unifi.sendSmartDetectEvent(this.motionEventId, 'stop', {
        className: decision.token,
        score: decision.scoreValue
      });
    }

    this.providers.unifi.sendMotionEvent(this.motionEventId, 'stop');
    this.motionEventId++;
  }

  shouldEmitSmartDetectEvent(decision) {
    if (!this.config?.detection?.emitSmartEvents) return false;
    return !!(decision?.token && decision.token !== '__unknown__');
  }

  logDetectionDecision(signal, decision) {
    if (!this.config?.detection?.logDecisions) return;

    const reasonCode = decision.reason || null;

    this.logger.debug('Detection gate decision', {
      phase: signal.phase || null,
      rawClassName: signal.className || null,
      token: decision.token,
      rawScore: signal.score ?? null,
      scoreValue: decision.scoreValue,
      minScore: decision.minScore,
      accepted: decision.accepted,
      reason: reasonCode,
      reasonLabel: getDecisionReasonLabel(reasonCode),
      transition: decision.transition,
      activeCountBefore: decision.activeCountBefore,
      activeCountAfter: decision.activeCountAfter
    });
  }

  async stop() {
    if (!this.isRunning) return;
    
    this.logger.info('Stopping camera proxy');
    this.isRunning = false;

    const components = [
      { name: 'unifi', component: this.providers.unifi },
      { name: 'video', component: this.providers.video },
      { name: 'rtspParity', component: this.services.rtspParity },
      { name: 'discovery', component: this.services.discovery },
      { name: 'onvif', component: this.providers.onvif }
    ];

    for (const { name, component } of components) {
      if (component) {
        try {
          this.logger.debug('Stopping ' + name);
          if (typeof component.stop === 'function') {
            await component.stop();
          } else if (typeof component.disconnect === 'function') {
            await component.disconnect();
          } else if (typeof component.shutdown === 'function') {
            await component.shutdown();
          }
        } catch (err) {
          this.logger.error('Error stopping ' + name, err);
        }
      }
    }

    this.logger.info('Camera proxy stopped');
  }
}

module.exports = { CameraProxy };
