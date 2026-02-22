# UniFi Camera Proxy

ONVIF to UniFi Protect camera proxy. Makes any ONVIF-compatible camera appear as a UniFi camera in Protect.

## Features

- **Discovery**: UDP multicast/broadcast discovery responder
- **ONVIF Integration**: Connects to ONVIF cameras using `agsh/onvif`
- **Video Streaming**: RTSP to FLV streaming via node-av or FFmpeg
- **WebSocket Protocol**: Full UniFi Protect AV client protocol
- **Motion Detection**: ONVIF events forwarded to UniFi
- **Auto SSL**: Automatic certificate generation
- **JSON Config**: File-based configuration

## Installation

```bash
npm install
# or
pnpm install
```

## Quick Start

### 1. Create Configuration File

```bash
npm run init-config
# Creates config.json with sample values
```

### 2. Edit Configuration

Edit `config.json` with your settings.

### 3. Run Proxy

```bash
sudo npm start
```

## Project Structure

```
camera-proxy/
├── bin/
│   └── unifi-camera-proxy      # CLI entry point
├── src/
│   ├── cli/                    # Command-line interface
│   ├── config/                 # Configuration management
│   ├── core/                   # Core orchestration (CameraProxy)
│   ├── domain/                 # Domain models
│   ├── providers/              # External service providers
│   │   ├── unifi/              # UniFi Protect provider
│   │   ├── onvif/              # ONVIF camera provider
│   │   └── video/              # Video streaming provider
│   └── services/               # Internal services
│       └── discovery.js        # UDP discovery service
├── config.json.example         # Example configuration
└── package.json
```

## Usage

See config.json.example for full configuration options.

## RTSP SmartFace Backend

Set `detection.rtspParity.enabled=true` and `detection.rtspParity.rtspUrl` to enable the Python parity runner.

Use `detection.rtspParity.pythonPath = "python3"` for production/runtime by default.

For SmartFace-assisted face events, use:

- `detection.rtspParity.detectorBackend = "smartface_ncnn"`
- `detection.rtspParity.smartfaceParam = "<path-to-SmartFace_extract_20220317.param>"`
- `detection.rtspParity.smartfaceBin = "<path-to-SmartFace_extract_20220317.bin>"`

The backend keeps person/vehicle inference and adds face detection with score gating and frame-stability filtering.

`ParityFrameSummary` now includes SmartFace tuning telemetry in `payload`:

- `faceScoreTelemetry.summary`: `count`, `min`, `max`, `mean`, `p50`, `p90`
- `faceScoreTelemetry.histogram`: fixed bins (`0.0-0.5`, `0.5-0.6`, `0.6-0.7`, `0.7-0.8`, `0.8-0.9`, `0.9-1.0`) and per-bin counts
- `stableFaceCount` and `smartfaceStableFramesThreshold`

Suggested tuning loop:

- If `faceCount > 0` frequently but `stableFaceCount` is often `0`, reduce `smartfaceStableFrames`.
- If `p90` stays below `smartfaceMinScore` on valid scenes, lower `smartfaceMinScore` gradually.
- Keep changes small and re-check histogram/percentile movement over representative day/night clips.

## Runtime vs Research Files

- Main project runtime code lives under `src/`, `bin/`, and `config.json.example`.
- Reverse-engineering and analysis harness artifacts live under `analysis/` and are research-only by default.
- Local experiments and reports can stay in `analysis/`, while production runtime should not depend on `analysis/` defaults.

## Operator Runbook

- **Runtime-only deploys**: package `src/`, `bin/`, `package.json`, and `config.json`; do not package `analysis/`.
- **Python runtime**: keep `detection.rtspParity.pythonPath = "python3"` unless you intentionally provide another interpreter.
- **Parity script path**: startup validates `detection.rtspParity.scriptPath` (or default `bin/rtsp-parity-runner.py`) exists; parity is disabled with a warning if the script file is missing.
- **SmartFace backend**: when `detection.rtspParity.detectorBackend = "smartface_ncnn"`, set both `smartfaceParam` and `smartfaceBin` to real model files accessible on the runtime host.
- **SmartFace preflight**: startup validates that `smartfaceParam` and `smartfaceBin` files exist; parity is disabled with a warning when either file is missing.
- **Research workflows**: keep captures, reverse scripts, and reports under `analysis/` so they stay separated from production runtime code.

## License

MIT
