const UNKNOWN_TOKEN = '__unknown__';

function mapDetectionClass(className) {
  if (typeof className !== 'string') return null;

  const value = className.trim().toLowerCase();
  if (!value) return null;

  if (value.includes('person') || value.includes('human')) return 'person';
  if (
    value.includes('vehicle') || value.includes('car') || value.includes('truck') ||
    value.includes('bus') || value.includes('motorcycle') || value.includes('bike')
  ) {
    return 'vehicle';
  }
  if (value.includes('animal') || value.includes('pet') || value.includes('dog') || value.includes('cat')) return 'animal';
  if (value.includes('face')) return 'face';
  if (value.includes('licenseplate') || value.includes('license_plate') || value.includes('plate') || value.includes('lpr')) {
    return 'licensePlate';
  }
  if (value.includes('package') || value.includes('parcel')) return 'package';

  return value;
}

class DetectionGate {
  constructor(config = {}) {
    const configuredClasses = Array.isArray(config.enabledClasses) ? config.enabledClasses : [];

    this.enabledClasses = new Set(
      configuredClasses
        .map((className) => mapDetectionClass(className))
        .filter(Boolean)
    );
    this.allowUnknownClass = config.allowUnknownClass !== false;
    this.minScore = config.minScore && typeof config.minScore === 'object' ? config.minScore : {};
    this.activeTokens = new Set();
  }

  isActive() {
    return this.activeTokens.size > 0;
  }

  apply(signal = {}) {
    return this.evaluate(signal).transition;
  }

  evaluate(signal = {}) {
    const phase = signal.phase === 'start' || signal.phase === 'stop' ? signal.phase : null;
    const scoreValue = this.resolveNumericScore(signal.score);
    const activeCountBefore = this.activeTokens.size;
    if (!phase) {
      return {
        accepted: false,
        reason: 'invalid_phase',
        transition: null,
        token: null,
        phase: null,
        minScore: null,
        scoreValue,
        activeCountBefore,
        activeCountAfter: activeCountBefore
      };
    }

    const token = this.getToken(signal.className);
    if (phase === 'start') {
      const startEligibility = this.evaluateStartEligibility(token, scoreValue);
      if (!startEligibility.allowed) {
        return {
          accepted: false,
          reason: startEligibility.reason,
          transition: null,
          token,
          phase,
          minScore: startEligibility.minScore,
          scoreValue,
          activeCountBefore,
          activeCountAfter: activeCountBefore
        };
      }
    }

    if (phase === 'stop' && token === UNKNOWN_TOKEN && !this.allowUnknownClass) {
      return {
        accepted: false,
        reason: 'unknown_class_blocked',
        transition: null,
        token,
        phase,
        minScore: null,
        scoreValue,
        activeCountBefore,
        activeCountAfter: activeCountBefore
      };
    }

    const wasActive = this.isActive();
    if (phase === 'start') {
      this.activeTokens.add(token);
    } else {
      this.activeTokens.delete(token);
    }

    const isActive = this.isActive();
    const transition = this.resolveTransition(wasActive, isActive);

    return {
      accepted: true,
      reason: this.resolveDecisionReason(phase, transition, wasActive, isActive),
      transition,
      token,
      phase,
      minScore: token === UNKNOWN_TOKEN ? null : this.resolveMinScore(token),
      scoreValue,
      activeCountBefore,
      activeCountAfter: this.activeTokens.size
    };
  }

  getToken(className) {
    const detectionClass = mapDetectionClass(className);
    return detectionClass || UNKNOWN_TOKEN;
  }

  evaluateStartEligibility(token, scoreValue) {
    if (token === UNKNOWN_TOKEN) {
      return {
        allowed: this.allowUnknownClass,
        reason: this.allowUnknownClass ? null : 'unknown_class_blocked',
        minScore: null
      };
    }

    if (this.enabledClasses.size > 0 && !this.enabledClasses.has(token)) {
      return {
        allowed: false,
        reason: 'class_not_enabled',
        minScore: null
      };
    }

    const minScore = this.resolveMinScore(token);
    if (scoreValue !== null && scoreValue < minScore) {
      return {
        allowed: false,
        reason: 'below_min_score',
        minScore
      };
    }

    return {
      allowed: true,
      reason: null,
      minScore
    };
  }

  resolveTransition(wasActive, isActive) {
    if (!wasActive && isActive) return 'start';
    if (wasActive && !isActive) return 'stop';
    return null;
  }

  resolveDecisionReason(phase, transition, wasActive, isActive) {
    if (transition === 'start') return 'gate_opened';
    if (transition === 'stop') return 'gate_closed';
    if (phase === 'start' && wasActive && isActive) return 'already_active';
    if (phase === 'stop' && wasActive && isActive) return 'still_active';
    if (phase === 'stop' && !wasActive && !isActive) return 'already_inactive';
    return 'no_transition';
  }

  resolveMinScore(className) {
    const perClass = this.minScore[className];
    if (typeof perClass === 'number') return perClass;

    const fallback = this.minScore.default;
    return typeof fallback === 'number' ? fallback : 0;
  }

  resolveNumericScore(score) {
    if (typeof score === 'number' && Number.isFinite(score)) return score;
    if (typeof score !== 'string') return null;

    const parsed = parseFloat(score);
    if (!Number.isFinite(parsed)) return null;
    return parsed;
  }
}

module.exports = { DetectionGate, mapDetectionClass };
