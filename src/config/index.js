const os = require('os');
const fs = require('fs');
const path = require('path');

function envInt(name, fallback) {
  const raw = process.env[name];
  if (raw === undefined || raw === null || raw === '') return fallback;
  const parsed = Number.parseInt(raw, 10);
  return Number.isFinite(parsed) ? parsed : fallback;
}

function envFloat(name, fallback) {
  const raw = process.env[name];
  if (raw === undefined || raw === null || raw === '') return fallback;
  const parsed = Number.parseFloat(raw);
  return Number.isFinite(parsed) ? parsed : fallback;
}

class ConfigManager {
  constructor() {
    this.defaults = {
      unifi: {
        host: process.env.UNIFI_HOST || '192.168.1.1',
        port: envInt('UNIFI_PORT', 7443),
        token: process.env.UNIFI_TOKEN || null,
        reconnect: {
          enabled: true,
          maxAttempts: 10,
          backoffMultiplier: 2,
          initialDelayMs: 1000,
          maxDelayMs: 30000
        }
      },
      camera: {
        mac: process.env.UNIFI_MAC || this.getDefaultMac(),
        ip: process.env.UNIFI_IP || this.getDefaultIp(),
        name: process.env.UNIFI_NAME || 'ONVIF Camera',
        model: process.env.UNIFI_MODEL || 'UVC G4 Bullet',
        firmware: process.env.UNIFI_FIRMWARE || '4.67.55'
      },
      onvif: {
        host: process.env.ONVIF_HOST || null,
        port: envInt('ONVIF_PORT', 80),
        username: process.env.ONVIF_USERNAME || 'admin',
        password: process.env.ONVIF_PASSWORD || 'admin',
        motionEnabled: process.env.MOTION_ENABLED !== 'false',
        motionPollInterval: envInt('MOTION_POLL_INTERVAL', 1000)
      },
      detection: {
        enabledClasses: ['person', 'vehicle', 'animal'],
        allowUnknownClass: true,
        emitSmartEvents: process.env.DETECTION_EMIT_SMART_EVENTS === 'true',
        logDecisions: process.env.DETECTION_LOG_DECISIONS === 'true',
        rtspParity: {
          enabled: process.env.RTSP_PARITY_ENABLED === 'true',
          pythonPath: process.env.RTSP_PARITY_PYTHON || null,
          scriptPath: process.env.RTSP_PARITY_SCRIPT || null,
          rtspUrl: process.env.RTSP_PARITY_RTSP_URL || null,
          detectorBackend: process.env.RTSP_PARITY_DETECTOR_BACKEND || 'heuristic',
          detectorModel: process.env.RTSP_PARITY_DETECTOR_MODEL || null,
          detectorConfThreshold: envFloat('RTSP_PARITY_DETECTOR_CONF_THRESHOLD', 0.35),
          detectorNmsThreshold: envFloat('RTSP_PARITY_DETECTOR_NMS_THRESHOLD', 0.45),
          detectorInputSize: envInt('RTSP_PARITY_DETECTOR_INPUT_SIZE', 640),
          smartfaceParam: process.env.RTSP_PARITY_SMARTFACE_PARAM || null,
          smartfaceBin: process.env.RTSP_PARITY_SMARTFACE_BIN || null,
          smartfaceInputSize: envInt('RTSP_PARITY_SMARTFACE_INPUT_SIZE', 112),
          smartfaceMinScore: envFloat('RTSP_PARITY_SMARTFACE_MIN_SCORE', 0.75),
          smartfaceStableFrames: envInt('RTSP_PARITY_SMARTFACE_STABLE_FRAMES', 2),
          maxFrames: envInt('RTSP_PARITY_MAX_FRAMES', 0),
          motionAreaRatioMin: envFloat('RTSP_PARITY_MOTION_AREA_RATIO_MIN', 0.004),
          sleepMs: envInt('RTSP_PARITY_SLEEP_MS', 0)
        },
        minScore: {
          person: 0.65,
          vehicle: 0.6,
          animal: 0.6,
          face: 0.75,
          licensePlate: 0.8,
          default: 0
        }
      },
      tls: {
        autoGenerate: true,
        cert: process.env.TLS_CERT_PATH || null,
        key: process.env.TLS_KEY_PATH || null
      },
      video: {
        useNodeAv: process.env.USE_NODE_AV !== 'false',
        fallbackToFfmpeg: true
      },
      logging: {
        level: process.env.LOG_LEVEL || 'info'
      }
    };
  }

  getDefaultMac() {
    const interfaces = os.networkInterfaces();
    for (const name of Object.keys(interfaces)) {
      for (const iface of interfaces[name]) {
        if (!iface.internal && iface.family === 'IPv4' && iface.mac) {
          return iface.mac;
        }
      }
    }
    return 'aa:bb:cc:dd:ee:ff';
  }

  getDefaultIp() {
    const interfaces = os.networkInterfaces();
    for (const name of Object.keys(interfaces)) {
      for (const iface of interfaces[name]) {
        if (!iface.internal && iface.family === 'IPv4' && iface.address) {
          return iface.address;
        }
      }
    }
    return '192.168.1.100';
  }

  load(configPath, cliOptions) {
    let config = JSON.parse(JSON.stringify(this.defaults));
    
    const searchPaths = [
      configPath,
      './config.json',
      path.join(os.homedir(), '.unifi-camera-proxy', 'config.json'),
      '/etc/unifi-camera-proxy/config.json'
    ].filter(Boolean);

    for (const searchPath of searchPaths) {
      if (fs.existsSync(searchPath)) {
        try {
          const data = fs.readFileSync(searchPath, 'utf8');
          const fileConfig = JSON.parse(data);
          config = this.mergeDeep(config, fileConfig);
          break;
        } catch (err) {
          console.warn('Failed to load config from', searchPath);
        }
      }
    }

    if (cliOptions) {
      if (cliOptions.host) config.unifi.host = cliOptions.host;
      if (cliOptions.token) config.unifi.token = cliOptions.token;
      if (cliOptions.mac) config.camera.mac = cliOptions.mac;
      if (cliOptions.onvifHost) config.onvif.host = cliOptions.onvifHost;
      if (cliOptions.onvifUser) config.onvif.username = cliOptions.onvifUser;
      if (cliOptions.onvifPass) config.onvif.password = cliOptions.onvifPass;
    }

    this.validate(config);
    return config;
  }

  mergeDeep(target, source) {
    const output = { ...target };
    if (this.isObject(target) && this.isObject(source)) {
      Object.keys(source).forEach(key => {
        if (this.isObject(source[key])) {
          if (!(key in target)) {
            Object.assign(output, { [key]: source[key] });
          } else {
            output[key] = this.mergeDeep(target[key], source[key]);
          }
        } else {
          Object.assign(output, { [key]: source[key] });
        }
      });
    }
    return output;
  }

  isObject(item) {
    return item && typeof item === 'object' && !Array.isArray(item);
  }

  validate(config) {
    const errors = [];
    if (!config.unifi.host) errors.push('unifi.host is required');
    if (!config.unifi.token) errors.push('unifi.token is required');
    if (!config.onvif.host) errors.push('onvif.host is required');
    if (!config.camera.mac) errors.push('camera.mac is required');
    
    if (errors.length > 0) {
      throw new Error('Configuration validation failed');
    }
  }

  static createSample() {
    return {
      unifi: {
        host: '192.168.1.1',
        port: 7443,
        token: 'YOUR_ADOPTION_TOKEN_HERE',
        reconnect: {
          enabled: true,
          maxAttempts: 10
        }
      },
      camera: {
        mac: 'AABBCCDDEEFF',
        ip: '192.168.1.100',
        name: 'My ONVIF Camera',
        model: 'UVC G4 Bullet',
        firmware: '4.67.55'
      },
      onvif: {
        host: '192.168.1.50',
        port: 80,
        username: 'admin',
        password: 'admin',
        motionEnabled: true,
        motionPollInterval: 1000
      },
      detection: {
        enabledClasses: ['person', 'vehicle', 'animal'],
        allowUnknownClass: true,
        emitSmartEvents: false,
        logDecisions: false,
        rtspParity: {
          enabled: false,
          pythonPath: null,
          scriptPath: null,
          rtspUrl: null,
          detectorBackend: 'heuristic',
          detectorModel: null,
          detectorConfThreshold: 0.35,
          detectorNmsThreshold: 0.45,
          detectorInputSize: 640,
          smartfaceParam: null,
          smartfaceBin: null,
          smartfaceInputSize: 112,
          smartfaceMinScore: 0.75,
          smartfaceStableFrames: 2,
          maxFrames: 0,
          motionAreaRatioMin: 0.004,
          sleepMs: 0
        },
        minScore: {
          person: 0.65,
          vehicle: 0.6,
          animal: 0.6,
          face: 0.75,
          licensePlate: 0.8,
          default: 0
        }
      },
      tls: {
        autoGenerate: true
      },
      video: {
        useNodeAv: true,
        fallbackToFfmpeg: true
      },
      logging: {
        level: 'info'
      }
    };
  }
}

module.exports = { ConfigManager };
