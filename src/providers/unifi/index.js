const WebSocket = require('ws');
const fs = require('fs');
const EventEmitter = require('events');

class UnifiProvider extends EventEmitter {
  constructor(config, logger) {
    super();
    this.config = config;
    this.logger = logger;
    this.ws = null;
    this.connected = false;
    this.messageId = 0;
    this.reconnectAttempts = 0;
    this.videoProvider = null;
  }

  async connect() {
    const uri = 'wss://' + this.config.host + ':' + this.config.port + '/camera/1.0/ws?token=' + this.config.token;
    
    const headers = {
      'camera-mac': this.config.mac,
      'camera-model': this.config.model,
      'firmware-version': this.config.firmware
    };

    let tlsContext = { rejectUnauthorized: false };
    if (this.config.tlsCert && this.config.tlsKey) {
      try {
        tlsContext.cert = fs.readFileSync(this.config.tlsCert);
        tlsContext.key = fs.readFileSync(this.config.tlsKey);
      } catch (err) {
        this.logger.warn('Failed to load TLS credentials');
      }
    }

    this.logger.info('Connecting to UniFi Protect', { host: this.config.host });

    this.ws = new WebSocket(uri, { headers, ...tlsContext });

    this.ws.on('open', () => this.handleOpen());
    this.ws.on('message', (data) => this.handleMessage(data));
    this.ws.on('close', (code, reason) => this.handleClose(code, reason));
    this.ws.on('error', (err) => this.handleError(err));
  }

  handleOpen() {
    this.logger.info('Connected to UniFi Protect');
    this.connected = true;
    this.reconnectAttempts = 0;
    this.sendHandshake();
  }

  async handleMessage(data) {
    try {
      const msg = JSON.parse(data);
      this.logger.debug('Received message', { function: msg.functionName });
      await this.processMessage(msg);
    } catch (err) {
      this.logger.error('Failed to parse message', err);
    }
  }

  async processMessage(msg) {
    const fn = msg.functionName;
    let response = null;

    switch (fn) {
      case 'ubnt_avclient_time':
        response = this.createResponse('ubnt_avclient_paramAgreement', msg.messageId, {
          monotonicMs: Math.floor(process.uptime() * 1000),
          wallMs: Date.now()
        });
        break;
      case 'ubnt_avclient_paramAgreement':
        response = this.createResponse('ubnt_avclient_paramAgreement', msg.messageId, {
          authToken: this.config.token,
          features: { mic: true, videoMode: ['default'], motionDetect: ['enhanced'] }
        });
        break;
      case 'ChangeVideoSettings':
        response = await this.handleVideoSettings(msg);
        break;
      case 'NetworkStatus':
        response = this.createResponse('NetworkStatus', msg.messageId, {
          connectionState: 2,
          connectionStateDescription: 'CONNECTED',
          ipAddress: this.config.ip
        });
        break;
      case 'ChangeDeviceSettings':
        response = this.createResponse('ChangeDeviceSettings', msg.messageId, {
          name: this.config.name
        });
        break;
    }

    if (response && msg.responseExpected !== false) {
      this.send(response);
    }
  }

  async handleVideoSettings(msg) {
    const payload = msg.payload;
    if (payload?.video && this.videoProvider) {
      for (const [streamId, streamConfig] of Object.entries(payload.video)) {
        if (streamConfig?.avSerializer) {
          const dest = streamConfig.avSerializer.destinations?.[0];
          if (dest && !dest.includes('/dev/null')) {
            const streamName = streamConfig.avSerializer.parameters?.streamName;
            const [host, port] = dest.replace('tcp://', '').split(':');
            await this.videoProvider.startStream(
              streamId,
              streamName,
              host,
              Number.parseInt(port, 10)
            );
          }
        }
      }
    }
    return this.createResponse('ChangeVideoSettings', msg.messageId, this.getVideoSettings());
  }

  handleClose(code, reason) {
    this.logger.info('Connection closed', { code, reason: reason?.toString() });
    this.connected = false;
    this.attemptReconnect();
  }

  handleError(err) {
    this.logger.error('Connection error', err);
  }

  attemptReconnect() {
    if (!this.config.reconnect?.enabled || this.reconnectAttempts >= this.config.reconnect.maxAttempts) {
      this.logger.error('Max reconnection attempts reached');
      return;
    }

    this.reconnectAttempts++;
    const delay = Math.min(
      this.config.reconnect.initialDelayMs * Math.pow(this.config.reconnect.backoffMultiplier, this.reconnectAttempts),
      this.config.reconnect.maxDelayMs
    );

    this.logger.info('Reconnecting', { attempt: this.reconnectAttempts, delay });
    setTimeout(() => this.connect().catch(() => {}), delay);
  }

  sendHandshake() {
    const msg = this.createResponse('ubnt_avclient_hello', 0, {
      adoptionCode: this.config.token,
      connectionHost: this.config.host,
      connectionSecurePort: this.config.port,
      fwVersion: this.config.firmware,
      hwrev: 19,
      ip: this.config.ip,
      mac: this.config.mac,
      model: this.config.model,
      name: this.config.name,
      protocolVersion: 67,
      uptime: Math.floor(process.uptime() * 1000),
      features: { mic: true, aec: [], videoMode: ['default'], motionDetect: ['enhanced'] }
    });
    this.send(msg);
  }

  createResponse(functionName, inResponseTo, payload) {
    return {
      from: 'ubnt_avclient',
      functionName: functionName,
      inResponseTo: inResponseTo,
      messageId: ++this.messageId,
      payload: payload || {},
      responseExpected: false,
      to: 'UniFiVideo'
    };
  }

  send(msg) {
    if (this.ws?.readyState === WebSocket.OPEN) {
      this.ws.send(JSON.stringify(msg));
    }
  }

  sendMotionEvent(eventId, state) {
    const payload = {
      clockMonotonic: Math.floor(process.uptime() * 1000),
      clockWall: Date.now(),
      edgeType: state,
      eventId: eventId,
      eventType: 'motion',
      levels: { '0': state === 'start' ? 47 : 49 }
    };
    this.send(this.createResponse('EventAnalytics', 0, payload));
  }

  sendSmartDetectEvent(eventId, state, detection = {}) {
    if (state !== 'start' && state !== 'stop') return;

    const className = typeof detection.className === 'string' ? detection.className : null;
    if (!className) return;

    const uptimeMs = Math.floor(process.uptime() * 1000);
    const wallMs = Date.now();
    const payload = {
      clockBestMonotonic: uptimeMs,
      clockBestWall: wallMs,
      clockMonotonic: uptimeMs,
      clockStream: uptimeMs,
      clockStreamRate: 1000,
      clockWall: wallMs,
      edgeType: state === 'start' ? 'enter' : 'leave',
      eventId: eventId,
      eventType: 'motion',
      levels: { '0': state === 'start' ? 47 : 49 },
      motionHeatmap: '',
      motionSnapshot: '',
      objectTypes: [className],
      smartDetectSnapshot: '',
      zonesStatus: { '0': 48 }
    };

    if (typeof detection.score === 'number' && Number.isFinite(detection.score)) {
      payload.confidence = detection.score;
    }

    this.send(this.createResponse('EventSmartDetect', 0, payload));
  }

  sendSmartDetectIdentityEvent(eventId, state, detection = {}) {
    if (state !== 'start' && state !== 'stop') return;

    const identityId =
      typeof detection.identityId === 'string' && detection.identityId.trim().length > 0
        ? detection.identityId.trim()
        : null;
    if (!identityId) return;

    const uptimeMs = Math.floor(process.uptime() * 1000);
    const wallMs = Date.now();
    const payload = {
      clockBestMonotonic: uptimeMs,
      clockBestWall: wallMs,
      clockMonotonic: uptimeMs,
      clockStream: uptimeMs,
      clockStreamRate: 1000,
      clockWall: wallMs,
      edgeType: state === 'start' ? 'enter' : 'leave',
      eventId,
      eventType: 'motion',
      levels: { '0': state === 'start' ? 47 : 49 },
      motionHeatmap: '',
      motionSnapshot: '',
      objectTypes: ['person'],
      smartDetectSnapshot: '',
      zonesStatus: { '0': 48 },
      identityId
    };

    if (typeof detection.score === 'number' && Number.isFinite(detection.score)) {
      payload.confidence = detection.score;
    }

    if (typeof detection.distance === 'number' && Number.isFinite(detection.distance)) {
      payload.distance = detection.distance;
    }

    this.send(this.createResponse('EventSmartDetectIdentity', 0, payload));
  }

  getVideoSettings() {
    return {
      audio: { bitRate: 32000, channels: 1, enabled: true, sampleRate: 11025, type: 'aac' },
      video: {
        video1: { enabled: true, fps: 15, height: 1080, width: 1920, type: 'h264' },
        video2: { enabled: true, fps: 15, height: 720, width: 1280, type: 'h264' },
        video3: { enabled: true, fps: 15, height: 360, width: 640, type: 'h264' }
      }
    };
  }

  setVideoProvider(provider) {
    this.videoProvider = provider;
  }

  isConnected() {
    return this.connected;
  }

  async disconnect() {
    if (this.ws) {
      this.ws.close();
      this.ws = null;
    }
    this.connected = false;
  }
}

module.exports = { UnifiProvider };
