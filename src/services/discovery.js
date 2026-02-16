const dgram = require('dgram');
const os = require('os');

const PROTOCOL = {
  VERSION: 0x01,
  CMD_SEARCH: 0x04,
  CMD_INFO: 0x06
};

const TLV_TYPES = {
  HWADDR: 0x01,
  PRIMARY_ADDRESS: 0x02,
  FWVERSION: 0x0a,
  HOSTNAME: 0x0b,
  PLATFORM: 0x0c,
  IPINFO: 0x0d,
  WEBUI: 0x0e,
  WMODE: 0x0f,
  MGMT_IS_DEFAULT: 0x12,
  MODEL_SHORT: 0x13,
  MODEL: 0x14,
  SYSTEM_ID: 0x16,
  UPTIME: 0x17,
  SUPPORT_UCP4: 0x1d
};

class DiscoveryService {
  constructor(cameraConfig, logger) {
    this.config = cameraConfig;
    this.logger = logger;
    this.socket = null;
    this.isRunning = false;
  }

  start() {
    this.socket = dgram.createSocket('udp4');

    this.socket.on('message', (msg, rinfo) => this.handleMessage(msg, rinfo));
    this.socket.on('error', (err) => {
      this.logger.error('Discovery socket error', err);
      this.stop();
    });

    this.socket.on('listening', () => {
      const address = this.socket.address();
      this.logger.info('Discovery service listening', { port: address.port });
      
      try {
        this.socket.addMembership('233.89.188.1');
        this.logger.debug('Joined multicast group');
      } catch (err) {
        this.logger.debug('Failed to join multicast (needs sudo)');
      }
    });

    this.socket.bind({ port: 10001, address: '0.0.0.0', reuseAddr: true });
    this.isRunning = true;
  }

  handleMessage(msg, rinfo) {
    if (msg.length < 4) return;

    const version = msg.readUInt8(0);
    const command = msg.readUInt8(1);

    if (version !== PROTOCOL.VERSION || command !== PROTOCOL.CMD_SEARCH) {
      return;
    }

    this.logger.debug('Discovery request received', { from: rinfo.address });

    const response = this.buildResponse();
    this.socket.send(response, rinfo.port, rinfo.address, (err) => {
      if (err) {
        this.logger.error('Failed to send discovery response', err);
      }
    });
  }

  buildResponse() {
    const mac = this.parseMac(this.config.mac);
    const ip = this.parseIp(this.config.ip);
    const tlvs = [];

    tlvs.push(this.createTlv(TLV_TYPES.PRIMARY_ADDRESS, Buffer.concat([mac, ip])));
    tlvs.push(this.createStringTlv(TLV_TYPES.HOSTNAME, this.config.name.replace(/ /g, '-')));
    tlvs.push(this.createStringTlv(TLV_TYPES.PLATFORM, 'UVC'));
    tlvs.push(this.createStringTlv(TLV_TYPES.FWVERSION, this.config.firmware));
    tlvs.push(this.createUint8Tlv(TLV_TYPES.MGMT_IS_DEFAULT, 1));
    tlvs.push(this.createStringTlv(TLV_TYPES.MODEL, this.config.model));
    tlvs.push(this.createStringTlv(TLV_TYPES.MODEL_SHORT, this.config.model.replace(/ /g, '-')));
    tlvs.push(this.createUint16LeTlv(TLV_TYPES.SYSTEM_ID, 0x0891));
    tlvs.push(this.createTlv(TLV_TYPES.IPINFO, Buffer.concat([mac, ip])));
    tlvs.push(this.createUint32BeTlv(TLV_TYPES.UPTIME, Math.floor(process.uptime())));
    tlvs.push(this.createTlv(TLV_TYPES.HWADDR, mac));
    tlvs.push(this.createUint8Tlv(TLV_TYPES.SUPPORT_UCP4, 1));

    const payloadLen = tlvs.reduce((sum, tlv) => sum + tlv.length, 0);
    const header = Buffer.alloc(4);
    header.writeUInt8(PROTOCOL.VERSION, 0);
    header.writeUInt8(PROTOCOL.CMD_INFO, 1);
    header.writeUInt16BE(payloadLen, 2);

    return Buffer.concat([header, Buffer.concat(tlvs)]);
  }

  parseMac(macStr) {
    return Buffer.from(macStr.replace(/:/g, '').match(/.{2}/g).map(b => parseInt(b, 16)));
  }

  parseIp(ipStr) {
    return Buffer.from(ipStr.split('.').map(n => parseInt(n, 10)));
  }

  createTlv(type, value) {
    return Buffer.concat([Buffer.from([type]), Buffer.alloc(2).fill(value.length), value]);
  }

  createStringTlv(type, str) {
    return this.createTlv(type, Buffer.from(str, 'ascii'));
  }

  createUint8Tlv(type, val) {
    return this.createTlv(type, Buffer.from([val]));
  }

  createUint16LeTlv(type, val) {
    const buf = Buffer.alloc(2);
    buf.writeUInt16LE(val, 0);
    return this.createTlv(type, buf);
  }

  createUint32BeTlv(type, val) {
    const buf = Buffer.alloc(4);
    buf.writeUInt32BE(val, 0);
    return this.createTlv(type, buf);
  }

  stop() {
    if (this.socket) {
      this.socket.close();
      this.socket = null;
    }
    this.isRunning = false;
    this.logger.info('Discovery service stopped');
  }
}

module.exports = { DiscoveryService };
