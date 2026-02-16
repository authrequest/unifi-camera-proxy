const { spawn } = require('child_process');
const EventEmitter = require('events');

let NodeAv;
try {
  NodeAv = require('node-av/api');
} catch {
  NodeAv = null;
}

class VideoProvider extends EventEmitter {
  constructor(config, logger) {
    super();
    this.config = config;
    this.logger = logger;
    this.streams = new Map();
    this.sourceUrl = null;
    this.useNodeAv = !!NodeAv && config.useNodeAv !== false;
  }

  async initialize() {
    this.logger.info('Initializing video provider', { useNodeAv: this.useNodeAv });
    if (!this.useNodeAv) {
      await this.validateFfmpeg();
    }
  }

  validateFfmpeg() {
    return new Promise((resolve, reject) => {
      const ffmpeg = spawn('ffmpeg', ['-version']);
      ffmpeg.on('error', () => reject(new Error('FFmpeg not found')));
      ffmpeg.on('close', (code) => {
        if (code === 0) resolve();
        else reject(new Error('FFmpeg validation failed'));
      });
    });
  }

  setSourceUrl(url) {
    this.sourceUrl = url;
  }

  async startStream(streamId, streamName, destHost, destPort) {
    if (!this.sourceUrl) {
      this.logger.error('No source URL set');
      return;
    }

    if (this.streams.has(streamId)) {
      this.logger.debug('Stream already running', { streamId });
      return;
    }

    this.logger.info('Starting stream', { streamId, streamName, dest: destHost + ':' + destPort });

    if (this.useNodeAv) {
      await this.startNativeStream(streamId, streamName, destHost, destPort);
    } else {
      await this.startFfmpegStream(streamId, streamName, destHost, destPort);
    }
  }

  async startNativeStream(streamId, streamName, destHost, destPort) {
    try {
      const { Demuxer, Decoder, Encoder, Muxer, HardwareContext } = NodeAv;
      
      const input = await Demuxer.open(this.sourceUrl);
      const videoStream = input.video();
      if (!videoStream) throw new Error('No video stream');

      const hw = HardwareContext.auto();
      const decoder = await Decoder.create(videoStream, { hardware: hw });
      const encoder = await Encoder.create('libx264', { decoder, bitrate: 2000000, fps: 15 });
      const output = await Muxer.open('tcp://' + destHost + ':' + destPort, { format: 'flv', input });
      const outputIndex = output.addStream(encoder);

      const context = { input, decoder, encoder, output, outputIndex, active: true };
      this.streams.set(streamId, context);
      this.processNativeStream(streamId, context);
    } catch (err) {
      this.logger.error('Native stream failed', err);
      if (this.config.fallbackToFfmpeg) {
        await this.startFfmpegStream(streamId, streamName, destHost, destPort);
      }
    }
  }

  async processNativeStream(streamId, context) {
    try {
      const { input, decoder, encoder, output, outputIndex } = context;
      const inputGen = input.packets(decoder.stream.index);
      const decoderGen = decoder.frames(inputGen);
      const encoderGen = encoder.packets(decoderGen);

      for await (const packet of encoderGen) {
        if (!context.active) break;
        await output.writePacket(packet, outputIndex);
      }
    } catch (err) {
      this.logger.error('Native stream error', err);
    } finally {
      this.streams.delete(streamId);
    }
  }

  async startFfmpegStream(streamId, streamName, destHost, destPort) {
    const args = [
      '-nostdin', '-loglevel', 'error', '-y',
      '-fflags', '+genpts+discardcorrupt',
      '-use_wallclock_as_timestamps', '1',
      '-rtsp_transport', 'tcp',
      '-i', this.sourceUrl,
      '-c:v', 'copy', '-c:a', 'aac',
      '-ar', '32000', '-ac', '1', '-b:a', '32000',
      '-f', 'flv', '-metadata', 'streamName=' + streamName,
      'tcp://' + destHost + ':' + destPort
    ];

    const ffmpeg = spawn('ffmpeg', args, { detached: false, windowsHide: true });
    
    ffmpeg.on('error', (err) => {
      this.logger.error('FFmpeg error', err);
      this.stopStream(streamId);
    });

    ffmpeg.on('close', () => {
      this.streams.delete(streamId);
    });

    this.streams.set(streamId, { process: ffmpeg, type: 'ffmpeg' });
  }

  async stopStream(streamId) {
    const stream = this.streams.get(streamId);
    if (!stream) return;

    if (stream.type === 'ffmpeg') {
      stream.process.kill('SIGTERM');
      setTimeout(() => {
        if (!stream.process.killed) stream.process.kill('SIGKILL');
      }, 5000);
    } else {
      stream.active = false;
      try {
        await stream.input?.close();
        await stream.output?.close();
      } catch (err) {
        this.logger.debug('Error closing native stream', err);
      }
    }

    this.streams.delete(streamId);
  }

  async shutdown() {
    for (const streamId of this.streams.keys()) {
      await this.stopStream(streamId);
    }
  }
}

module.exports = { VideoProvider };
