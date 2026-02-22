const path = require('path');
const { spawn } = require('child_process');
const readline = require('readline');

class RtspParityBridge {
  constructor(config = {}, logger) {
    this.config = config;
    this.logger = logger;
    this.proc = null;
    this.stdoutInterface = null;
    this.stderrInterface = null;
  }

  buildCommand() {
    const pythonPath =
      typeof this.config.pythonPath === 'string' && this.config.pythonPath
        ? this.config.pythonPath
        : 'python3';

    const scriptPath =
      typeof this.config.scriptPath === 'string' && this.config.scriptPath
        ? this.config.scriptPath
        : path.resolve(process.cwd(), 'bin/rtsp-parity-runner.py');

    const args = [
      scriptPath,
      '--rtsp-url',
      String(this.config.rtspUrl || '')
    ];

    if (typeof this.config.maxFrames === 'number' && this.config.maxFrames > 0) {
      args.push('--max-frames', String(this.config.maxFrames));
    }

    if (
      typeof this.config.motionAreaRatioMin === 'number' &&
      Number.isFinite(this.config.motionAreaRatioMin) &&
      this.config.motionAreaRatioMin > 0
    ) {
      args.push('--motion-area-ratio-min', String(this.config.motionAreaRatioMin));
    }

    if (typeof this.config.sleepMs === 'number' && this.config.sleepMs > 0) {
      args.push('--sleep-ms', String(this.config.sleepMs));
    }

    if (typeof this.config.detectorBackend === 'string' && this.config.detectorBackend) {
      args.push('--detector-backend', this.config.detectorBackend);
    }

    if (typeof this.config.detectorModel === 'string' && this.config.detectorModel) {
      args.push('--detector-model', this.config.detectorModel);
    }

    if (
      typeof this.config.detectorConfThreshold === 'number' &&
      Number.isFinite(this.config.detectorConfThreshold) &&
      this.config.detectorConfThreshold >= 0
    ) {
      args.push('--detector-conf-threshold', String(this.config.detectorConfThreshold));
    }

    if (
      typeof this.config.detectorNmsThreshold === 'number' &&
      Number.isFinite(this.config.detectorNmsThreshold) &&
      this.config.detectorNmsThreshold >= 0
    ) {
      args.push('--detector-nms-threshold', String(this.config.detectorNmsThreshold));
    }

    if (typeof this.config.detectorInputSize === 'number' && this.config.detectorInputSize > 0) {
      args.push('--detector-input-size', String(this.config.detectorInputSize));
    }

    if (typeof this.config.smartfaceParam === 'string' && this.config.smartfaceParam) {
      args.push('--smartface-param', this.config.smartfaceParam);
    }

    if (typeof this.config.smartfaceBin === 'string' && this.config.smartfaceBin) {
      args.push('--smartface-bin', this.config.smartfaceBin);
    }

    if (typeof this.config.smartfaceInputSize === 'number' && this.config.smartfaceInputSize > 0) {
      args.push('--smartface-input-size', String(this.config.smartfaceInputSize));
    }

    if (
      typeof this.config.smartfaceMinScore === 'number' &&
      Number.isFinite(this.config.smartfaceMinScore) &&
      this.config.smartfaceMinScore >= 0
    ) {
      args.push('--smartface-min-score', String(this.config.smartfaceMinScore));
    }

    if (
      typeof this.config.smartfaceStableFrames === 'number' &&
      Number.isFinite(this.config.smartfaceStableFrames) &&
      this.config.smartfaceStableFrames > 0
    ) {
      args.push('--smartface-stable-frames', String(this.config.smartfaceStableFrames));
    }

    return { pythonPath, args };
  }

  start(onMessage) {
    if (!this.config.rtspUrl) {
      throw new Error('rtspUrl is required for RTSP parity bridge');
    }

    if (typeof onMessage !== 'function') {
      throw new Error('onMessage callback is required for RTSP parity bridge');
    }

    if (this.proc) return;

    const { pythonPath, args } = this.buildCommand();
    this.proc = spawn(pythonPath, args, { stdio: ['ignore', 'pipe', 'pipe'] });

    this.stdoutInterface = readline.createInterface({ input: this.proc.stdout });
    this.stderrInterface = readline.createInterface({ input: this.proc.stderr });

    this.stdoutInterface.on('line', (line) => this.handleStdoutLine(line, onMessage));
    this.stderrInterface.on('line', (line) => {
      const text = String(line || '').trim();
      if (!text) return;
      this.logger.debug('RTSP parity stderr', { line: text });
    });

    this.proc.on('error', (err) => {
      this.logger.error('RTSP parity process error', err);
      this.cleanup();
    });

    this.proc.on('exit', (code, signal) => {
      this.logger.info('RTSP parity process exited', {
        code,
        signal: signal || null
      });
      this.cleanup();
    });

    this.logger.info('RTSP parity process started', {
      pythonPath,
      scriptPath: args[0]
    });
  }

  handleStdoutLine(line, onMessage) {
    const text = String(line || '').trim();
    if (!text) return;

    try {
      const parsed = JSON.parse(text);
      onMessage(parsed);
    } catch (_err) {
      this.logger.debug('RTSP parity non-json output', { line: text });
    }
  }

  stop() {
    if (this.stdoutInterface) {
      this.stdoutInterface.removeAllListeners();
      this.stdoutInterface.close();
      this.stdoutInterface = null;
    }

    if (this.stderrInterface) {
      this.stderrInterface.removeAllListeners();
      this.stderrInterface.close();
      this.stderrInterface = null;
    }

    if (this.proc) {
      this.proc.removeAllListeners();
      if (!this.proc.killed) {
        this.proc.kill('SIGTERM');
      }
      this.proc = null;
    }
  }

  cleanup() {
    if (this.stdoutInterface) {
      this.stdoutInterface.removeAllListeners();
      this.stdoutInterface.close();
      this.stdoutInterface = null;
    }

    if (this.stderrInterface) {
      this.stderrInterface.removeAllListeners();
      this.stderrInterface.close();
      this.stderrInterface = null;
    }

    this.proc = null;
  }
}

module.exports = { RtspParityBridge };
