const { program } = require('commander');
const winston = require('winston');
const fs = require('fs');

const { ConfigManager } = require('../config');
const { CameraProxy } = require('../core');
const { version } = require('../../package.json');

program
  .name('unifi-camera-proxy')
  .description('ONVIF to UniFi Protect camera proxy')
  .version(version);

program
  .option('-c, --config <path>', 'Configuration file path')
  .option('--init-config <path>', 'Create sample configuration file')
  .option('--host <host>', 'UniFi Protect NVR host address')
  .option('--token <token>', 'Adoption token from UniFi Protect UI')
  .option('--mac <mac>', 'Camera MAC address')
  .option('--onvif-host <host>', 'ONVIF camera host address')
  .option('--onvif-user <user>', 'ONVIF username')
  .option('--onvif-pass <pass>', 'ONVIF password')
  .option('-v, --verbose', 'Enable verbose logging')
  .option('--dry-run', 'Validate configuration only');

program.parse();

const cliOptions = program.opts();

if (cliOptions.initConfig) {
  const sample = ConfigManager.createSample();
  fs.writeFileSync(cliOptions.initConfig, JSON.stringify(sample, null, 2));
  console.log('Sample configuration created:', cliOptions.initConfig);
  process.exit(0);
}

async function main() {
  const configManager = new ConfigManager();
  const config = configManager.load(cliOptions.config, cliOptions);
  
  const logger = winston.createLogger({
    level: config.logging?.level || (cliOptions.verbose ? 'debug' : 'info'),
    format: winston.format.combine(
      winston.format.timestamp(),
      winston.format.colorize(),
      winston.format.printf(({ level, message, timestamp, ...meta }) => {
        let msg = `${timestamp} [${level}]: ${message}`;
        if (Object.keys(meta).length > 0) {
          msg += ` ${JSON.stringify(meta)}`;
        }
        return msg;
      })
    ),
    transports: [new winston.transports.Console()]
  });

  logger.info('Starting UniFi Camera Proxy');

  if (cliOptions.dryRun) {
    logger.info('Configuration validated successfully');
    logger.info('Config:', {
      unifi_host: config.unifi?.host,
      camera_mac: config.camera?.mac,
      onvif_host: config.onvif?.host
    });
    return;
  }

  const proxy = new CameraProxy(config, logger);

  ['SIGINT', 'SIGTERM'].forEach(signal => {
    process.on(signal, async () => {
      logger.info('Received ' + signal + ', shutting down...');
      await proxy.stop();
      process.exit(0);
    });
  });

  process.on('uncaughtException', (err) => {
    logger.error('Uncaught exception:', err);
    proxy.stop().then(() => process.exit(1));
  });

  try {
    await proxy.start();
  } catch (err) {
    logger.error('Failed to start proxy:', err);
    process.exit(1);
  }
}

main();
