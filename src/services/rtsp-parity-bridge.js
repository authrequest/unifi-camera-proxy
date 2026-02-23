const path = require('path');
const fs = require('fs');
const { spawn } = require('child_process');
const readline = require('readline');

class RtspParityBridge {
  constructor(config = {}, logger) {
    this.config = config;
    this.logger = logger;
    this.proc = null;
    this.stdoutInterface = null;
    this.stderrInterface = null;
    this.parityLogPath = null;
    this.parityLogStream = null;
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

    if (typeof this.config.smartfaceLmkParam === 'string' && this.config.smartfaceLmkParam) {
      args.push('--smartface-lmk-param', this.config.smartfaceLmkParam);
    }

    if (typeof this.config.smartfaceLmkBin === 'string' && this.config.smartfaceLmkBin) {
      args.push('--smartface-lmk-bin', this.config.smartfaceLmkBin);
    }

    if (typeof this.config.smartfaceExtractParam === 'string' && this.config.smartfaceExtractParam) {
      args.push('--smartface-extract-param', this.config.smartfaceExtractParam);
    }

    if (typeof this.config.smartfaceExtractBin === 'string' && this.config.smartfaceExtractBin) {
      args.push('--smartface-extract-bin', this.config.smartfaceExtractBin);
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

    if (
      typeof this.config.smartfaceIdentityDistanceThreshold === 'number' &&
      Number.isFinite(this.config.smartfaceIdentityDistanceThreshold) &&
      this.config.smartfaceIdentityDistanceThreshold >= 0
    ) {
      args.push(
        '--smartface-identity-distance-threshold',
        String(this.config.smartfaceIdentityDistanceThreshold)
      );
    }

    if (
      typeof this.config.smartfaceIdentityStableFrames === 'number' &&
      Number.isFinite(this.config.smartfaceIdentityStableFrames) &&
      this.config.smartfaceIdentityStableFrames > 0
    ) {
      args.push(
        '--smartface-identity-stable-frames',
        String(this.config.smartfaceIdentityStableFrames)
      );
    }

    if (
      typeof this.config.smartfaceIdentityMinFaceScore === 'number' &&
      Number.isFinite(this.config.smartfaceIdentityMinFaceScore) &&
      this.config.smartfaceIdentityMinFaceScore >= 0
    ) {
      args.push(
        '--smartface-identity-min-face-score',
        String(this.config.smartfaceIdentityMinFaceScore)
      );
    }

    if (
      typeof this.config.smartfaceIdentityMinFaceAreaRatio === 'number' &&
      Number.isFinite(this.config.smartfaceIdentityMinFaceAreaRatio) &&
      this.config.smartfaceIdentityMinFaceAreaRatio >= 0
    ) {
      args.push(
        '--smartface-identity-min-face-area-ratio',
        String(this.config.smartfaceIdentityMinFaceAreaRatio)
      );
    }

    if (this.config.smartfaceIdentityRequireLandmarks === true) {
      args.push('--smartface-identity-require-landmarks');
    }

    if (
      typeof this.config.smartfaceIdentitySplitGuardRatio === 'number' &&
      Number.isFinite(this.config.smartfaceIdentitySplitGuardRatio) &&
      this.config.smartfaceIdentitySplitGuardRatio >= 0
    ) {
      args.push(
        '--smartface-identity-split-guard-ratio',
        String(this.config.smartfaceIdentitySplitGuardRatio)
      );
    }

    if (
      typeof this.config.smartfaceIdentitySplitGuardMaxSeen === 'number' &&
      Number.isFinite(this.config.smartfaceIdentitySplitGuardMaxSeen) &&
      this.config.smartfaceIdentitySplitGuardMaxSeen >= 0
    ) {
      args.push(
        '--smartface-identity-split-guard-max-seen',
        String(this.config.smartfaceIdentitySplitGuardMaxSeen)
      );
    }

    if (
      typeof this.config.smartfaceIdentityMergeRecoverThreshold === 'number' &&
      Number.isFinite(this.config.smartfaceIdentityMergeRecoverThreshold) &&
      this.config.smartfaceIdentityMergeRecoverThreshold >= 0
    ) {
      args.push(
        '--smartface-identity-merge-recover-threshold',
        String(this.config.smartfaceIdentityMergeRecoverThreshold)
      );
    }

    if (
      typeof this.config.smartfaceIdentityMergeRecoverMinSeen === 'number' &&
      Number.isFinite(this.config.smartfaceIdentityMergeRecoverMinSeen) &&
      this.config.smartfaceIdentityMergeRecoverMinSeen > 0
    ) {
      args.push(
        '--smartface-identity-merge-recover-min-seen',
        String(this.config.smartfaceIdentityMergeRecoverMinSeen)
      );
    }

    if (this.config.smartfaceIdentityPreventDuplicatePerFrame === false) {
      args.push('--smartface-identity-allow-duplicate-per-frame');
    }

    if (this.config.smartfaceIdentityVerifiedAllowlistMode === true) {
      args.push('--smartface-identity-verified-allowlist-mode');
    }

    if (this.config.smartfaceIdentityQualityVerifierEnabled === true) {
      args.push('--smartface-identity-quality-verifier-enabled');
    }

    if (this.config.smartfaceIdentityPoseMaskGateEnabled === true) {
      args.push('--smartface-identity-pose-mask-gate-enabled');
    }

    if (
      typeof this.config.smartfaceIdentityMaxAbsYawDeg === 'number' &&
      Number.isFinite(this.config.smartfaceIdentityMaxAbsYawDeg) &&
      this.config.smartfaceIdentityMaxAbsYawDeg >= 0
    ) {
      args.push(
        '--smartface-identity-max-abs-yaw-deg',
        String(this.config.smartfaceIdentityMaxAbsYawDeg)
      );
    }

    if (
      typeof this.config.smartfaceIdentityMaxAbsPitchDeg === 'number' &&
      Number.isFinite(this.config.smartfaceIdentityMaxAbsPitchDeg) &&
      this.config.smartfaceIdentityMaxAbsPitchDeg >= 0
    ) {
      args.push(
        '--smartface-identity-max-abs-pitch-deg',
        String(this.config.smartfaceIdentityMaxAbsPitchDeg)
      );
    }

    if (
      typeof this.config.smartfaceIdentityMaxAbsRollDeg === 'number' &&
      Number.isFinite(this.config.smartfaceIdentityMaxAbsRollDeg) &&
      this.config.smartfaceIdentityMaxAbsRollDeg >= 0
    ) {
      args.push(
        '--smartface-identity-max-abs-roll-deg',
        String(this.config.smartfaceIdentityMaxAbsRollDeg)
      );
    }

    if (
      typeof this.config.smartfaceIdentityMaskConfidenceMin === 'number' &&
      Number.isFinite(this.config.smartfaceIdentityMaskConfidenceMin) &&
      this.config.smartfaceIdentityMaskConfidenceMin >= 0
    ) {
      args.push(
        '--smartface-identity-mask-confidence-min',
        String(this.config.smartfaceIdentityMaskConfidenceMin)
      );
    }

    if (typeof this.config.identityGalleryPath === 'string' && this.config.identityGalleryPath) {
      args.push('--identity-gallery-path', this.config.identityGalleryPath);
    }

    if (
      typeof this.config.identityGalleryMaxProfiles === 'number' &&
      Number.isFinite(this.config.identityGalleryMaxProfiles) &&
      this.config.identityGalleryMaxProfiles > 0
    ) {
      args.push(
        '--identity-gallery-max-profiles',
        String(this.config.identityGalleryMaxProfiles)
      );
    }

    if (
      typeof this.config.identityGallerySaveIntervalFrames === 'number' &&
      Number.isFinite(this.config.identityGallerySaveIntervalFrames) &&
      this.config.identityGallerySaveIntervalFrames > 0
    ) {
      args.push(
        '--identity-gallery-save-interval-frames',
        String(this.config.identityGallerySaveIntervalFrames)
      );
    }

    if (
      typeof this.config.identityGalleryMaxIdleMs === 'number' &&
      Number.isFinite(this.config.identityGalleryMaxIdleMs) &&
      this.config.identityGalleryMaxIdleMs >= 0
    ) {
      args.push(
        '--identity-gallery-max-idle-ms',
        String(this.config.identityGalleryMaxIdleMs)
      );
    }

    if (
      typeof this.config.identityGalleryPruneIntervalFrames === 'number' &&
      Number.isFinite(this.config.identityGalleryPruneIntervalFrames) &&
      this.config.identityGalleryPruneIntervalFrames > 0
    ) {
      args.push(
        '--identity-gallery-prune-interval-frames',
        String(this.config.identityGalleryPruneIntervalFrames)
      );
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

    this.openParityLogStream();

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
      scriptPath: args[0],
      parityLogPath: this.parityLogPath
    });
  }

  openParityLogStream() {
    const configuredPath =
      typeof this.config.parityLogPath === 'string'
        ? this.config.parityLogPath.trim()
        : '';
    if (!configuredPath) return;

    const resolvedPath = path.isAbsolute(configuredPath)
      ? configuredPath
      : path.resolve(process.cwd(), configuredPath);

    try {
      fs.mkdirSync(path.dirname(resolvedPath), { recursive: true });
      this.parityLogStream = fs.createWriteStream(resolvedPath, { flags: 'a' });
      this.parityLogPath = resolvedPath;

      this.parityLogStream.on('error', (err) => {
        this.logger.warn('RTSP parity log stream error', {
          parityLogPath: resolvedPath,
          error: err?.message || String(err)
        });
      });
    } catch (err) {
      this.parityLogPath = null;
      this.parityLogStream = null;
      this.logger.warn('Unable to open RTSP parity log file', {
        parityLogPath: resolvedPath,
        error: err?.message || String(err)
      });
    }
  }

  appendParityLogEntry(message) {
    if (!this.parityLogStream || !message || typeof message !== 'object') return;

    try {
      const line = JSON.stringify(message);
      this.parityLogStream.write(`${line}\n`);
    } catch (err) {
      this.logger.warn('Failed to append RTSP parity log entry', {
        parityLogPath: this.parityLogPath,
        error: err?.message || String(err)
      });
    }
  }

  closeParityLogStream() {
    if (!this.parityLogStream) {
      this.parityLogPath = null;
      return;
    }

    const stream = this.parityLogStream;
    this.parityLogStream = null;
    this.parityLogPath = null;
    stream.removeAllListeners('error');
    stream.end();
  }

  handleStdoutLine(line, onMessage) {
    const text = String(line || '').trim();
    if (!text) return;

    try {
      const parsed = JSON.parse(text);
      this.appendParityLogEntry(parsed);
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

    this.closeParityLogStream();
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
    this.closeParityLogStream();
  }
}

module.exports = { RtspParityBridge };
