const CLASS_KEY_NAMES = new Set(['objectclass', 'objecttype', 'class', 'type', 'label']);
const SCORE_KEY_NAMES = new Set(['confidence', 'score', 'probability', 'likelihood']);

const { mapDetectionClass } = require('./detection-gate');

function extractDetectionMetadata(simpleItems) {
  if (!Array.isArray(simpleItems)) {
    return { className: null, score: null };
  }

  let className = null;
  let score = null;

  for (const item of simpleItems) {
    const name = item?.$?.Name;
    const value = item?.$?.Value;
    if (typeof name !== 'string') continue;

    const fieldKey = name.toLowerCase();
    if (className === null && CLASS_KEY_NAMES.has(fieldKey)) {
      className = mapDetectionClass(value);
      continue;
    }

    if (score === null && SCORE_KEY_NAMES.has(fieldKey)) {
      score = parseConfidenceScore(value);
    }
  }

  return { className, score };
}

function parseConfidenceScore(value) {
  let numeric = null;

  if (typeof value === 'number' && Number.isFinite(value)) {
    numeric = value;
  } else if (typeof value === 'string') {
    const parsed = parseFloat(value);
    if (Number.isFinite(parsed)) {
      numeric = parsed;
    }
  }

  if (numeric === null) return null;
  if (numeric > 1 && numeric <= 100) return numeric / 100;
  if (numeric > 100) return 1;
  if (numeric < 0) return 0;

  return numeric;
}

module.exports = { extractDetectionMetadata, parseConfidenceScore };
