const { execSync } = require('child_process');
const fs = require('fs');
const path = require('path');
const crypto = require('crypto');

class TlsCertificate {
  constructor(config, logger) {
    this.config = config;
    this.logger = logger;
    this.certPath = null;
    this.keyPath = null;
  }

  ensureExists(baseDir) {
    if (this.config.cert && this.config.key) {
      if (fs.existsSync(this.config.cert) && fs.existsSync(this.config.key)) {
        this.certPath = this.config.cert;
        this.keyPath = this.config.key;
        this.logger.info('Using existing TLS certificate');
        return { cert: this.certPath, key: this.keyPath };
      }
    }

    if (this.config.autoGenerate) {
      return this.generate(baseDir);
    }

    throw new Error('TLS certificates not found and autoGenerate is disabled');
  }

  generate(baseDir) {
    this.certPath = path.join(baseDir, 'cert.pem');
    this.keyPath = path.join(baseDir, 'key.pem');

    if (fs.existsSync(this.certPath) && fs.existsSync(this.keyPath)) {
      this.logger.info('Using existing generated certificates');
      return { cert: this.certPath, key: this.keyPath };
    }

    this.logger.info('Generating self-signed TLS certificate');

    try {
      const subject = '/C=US/ST=California/L=San Francisco/O=UniFi Camera/CN=unifi.camera.local';
      const cmd = 'openssl req -x509 -newkey rsa:4096 ' +
        '-keyout "' + this.keyPath + '" ' +
        '-out "' + this.certPath + '" ' +
        '-days 3650 -nodes -subj "' + subject + '" ' +
        '-addext "subjectAltName=DNS:unifi.camera.local,DNS:localhost,IP:127.0.0.1"';

      execSync(cmd, { stdio: 'ignore' });
      fs.chmodSync(this.keyPath, 0o600);
      fs.chmodSync(this.certPath, 0o644);

      const fingerprint = this.getFingerprint();
      this.logger.info('Certificate generated', { fingerprint });
    } catch (err) {
      this.logger.error('Failed to generate certificate', err);
      throw err;
    }

    return { cert: this.certPath, key: this.keyPath };
  }

  getFingerprint() {
    try {
      const cert = fs.readFileSync(this.certPath);
      const hash = crypto.createHash('sha256').update(cert).digest('hex');
      return hash.match(/.{2}/g).join(':').toUpperCase();
    } catch {
      return null;
    }
  }
}

module.exports = { TlsCertificate };
