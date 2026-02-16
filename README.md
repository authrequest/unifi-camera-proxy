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

## License

MIT
