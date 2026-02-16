const os = require('os');
const fs = require('fs');
const path = require('path');

class ConfigManager {
  constructor() {
    this.defaults = {
      unifi: {
        host: process.env.UNIFI_HOST || '192.168.1.1',
        port: parseInt(process.env.UNIFI_PORT) || 7443,
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
        port: parseInt(process.env.ONVIF_PORT) || 80,
        username: process.env.ONVIF_USERNAME || 'admin',
        password: process.env.ONVIF_PASSWORD || 'admin',
        motionEnabled: process.env.MOTION_ENABLED !== 'false',
        motionPollInterval: parseInt(process.env.MOTION_POLL_INTERVAL) || 1000
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
