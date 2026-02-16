const { UnifiProvider } = require('../providers/unifi');
const { OnvifProvider } = require('../providers/onvif');
const { VideoProvider } = require('../providers/video');
const { DiscoveryService } = require('../services/discovery');
const { TlsCertificate } = require('../domain/tls-certificate');

class CameraProxy {
  constructor(config, logger) {
    this.config = config;
    this.logger = logger;
    this.providers = {};
    this.services = {};
    this.resources = {};
    this.isRunning = false;
    this.motionEventId = 0;
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

    this.providers.onvif.on('motion:start', () => this.handleMotionStart());
    this.providers.onvif.on('motion:stop', () => this.handleMotionStop());

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

    this.isRunning = true;
    this.logger.info('Camera proxy started successfully');
  }

  handleMotionStart() {
    if (!this.providers.unifi.isConnected()) return;
    
    this.logger.debug('Motion detected');
    this.providers.unifi.sendMotionEvent(this.motionEventId, 'start');
  }

  handleMotionStop() {
    if (!this.providers.unifi.isConnected()) return;
    
    this.logger.debug('Motion cleared');
    this.providers.unifi.sendMotionEvent(this.motionEventId, 'stop');
    this.motionEventId++;
  }

  async stop() {
    if (!this.isRunning) return;
    
    this.logger.info('Stopping camera proxy');
    this.isRunning = false;

    const components = [
      { name: 'unifi', component: this.providers.unifi },
      { name: 'video', component: this.providers.video },
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
