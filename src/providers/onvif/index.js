const onvif = require('onvif/promises');
const EventEmitter = require('events');
const { extractDetectionMetadata } = require('../../services/onvif-detection-parser');

class OnvifProvider extends EventEmitter {
  constructor(config, logger) {
    super();
    this.config = config;
    this.logger = logger;
    this.device = null;
    this.isConnected = false;
    this.streamUrl = null;
    this.motionPollInterval = null;
    this.motionState = false;
  }

  async connect() {
    if (!this.config.host) {
      throw new Error('ONVIF host is required');
    }

    this.logger.info('Connecting to ONVIF device', { host: this.config.host });

    this.device = new onvif.Cam({
      hostname: this.config.host,
      port: this.config.port,
      username: this.config.username,
      password: this.config.password,
      timeout: 10000
    });

    await this.device.connect();
    this.isConnected = true;

    this.logger.info('ONVIF device connected');

    await this.loadStreamUrl();

    if (this.config.motionEnabled !== false) {
      this.setupMotionDetection();
    }
  }

  async loadStreamUrl() {
    try {
      const uri = await this.device.getStreamUri({ protocol: 'RTSP', stream: 'RTP-Unicast' });
      let url = uri.uri;
      if (this.config.username && this.config.password) {
        url = url.replace('://', '://' + this.config.username + ':' + this.config.password + '@');
      }
      this.streamUrl = url;
      this.logger.info('Stream URL loaded');
    } catch (err) {
      this.logger.error('Failed to load stream URL', err);
    }
  }

  setupMotionDetection() {
    this.device.on('event', (event) => this.handleEvent(event));
    this.device.on('eventsError', (err) => this.logger.error('ONVIF event error', err));

    this.motionPollInterval = setInterval(async () => {
      try {
        await this.device.eventsPull();
      } catch (err) {
        this.logger.debug('Motion poll error', err.message);
      }
    }, this.config.motionPollInterval || 1000);

    this.logger.info('Motion detection enabled');
  }

  handleEvent(event) {
    const topic = event.topic?.name || '';
    if (!topic.includes('Motion') && !topic.includes('RuleEngine')) return;

    const data = event.message?.message?.data?.simpleItem || [];
    const stateItem = data.find(item => item.$?.Name === 'State');
    if (!stateItem) return;

    const newState = stateItem.$.Value === 'true' || stateItem.$.Value === true;
    if (newState !== this.motionState) {
      this.motionState = newState;
      const detectionMetadata = extractDetectionMetadata(data);
      if (newState) {
        this.emit('motion:start', detectionMetadata);
      } else {
        this.emit('motion:stop', detectionMetadata);
      }
    }
  }

  getStreamUrl() {
    return this.streamUrl;
  }

  async disconnect() {
    if (this.motionPollInterval) {
      clearInterval(this.motionPollInterval);
    }
    this.isConnected = false;
    this.logger.info('ONVIF device disconnected');
  }
}

module.exports = { OnvifProvider };
