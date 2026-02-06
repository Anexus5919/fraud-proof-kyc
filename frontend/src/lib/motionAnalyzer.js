/**
 * Motion Analyzer for Liveness Detection
 *
 * Analyzes face motion patterns to detect static photos/screens:
 * 1. Real faces have natural micro-movements (involuntary)
 * 2. Photos shown on phones have parallel motion (entire image moves together)
 * 3. Static photos have no motion at all
 *
 * This provides a client-side signal that complements server-side anti-spoofing.
 */

// Configuration
const CONFIG = {
  FRAME_HISTORY: 30,           // Number of frames to analyze
  MIN_FRAMES_FOR_ANALYSIS: 15, // Minimum frames needed
  MICRO_MOVEMENT_THRESHOLD: 0.001, // Minimum expected micro-movement
  PARALLEL_MOTION_THRESHOLD: 0.85, // Correlation threshold for detecting parallel motion
  STILLNESS_THRESHOLD: 0.0005,     // Below this is suspiciously still
};

// Key landmark indices for motion tracking
// Using a subset of MediaPipe Face Landmarker points
const TRACKED_LANDMARKS = {
  noseTip: 1,
  leftEyeOuter: 33,
  rightEyeOuter: 263,
  leftMouthCorner: 61,
  rightMouthCorner: 291,
  chin: 152,
  forehead: 10,
  leftCheek: 234,
  rightCheek: 454,
};

class MotionAnalyzer {
  constructor() {
    this.frameHistory = [];
    this.lastAnalysis = null;
  }

  /**
   * Add a new frame of landmarks for analysis
   * @param {Array} landmarks - Face landmarks from MediaPipe
   */
  addFrame(landmarks) {
    if (!landmarks || landmarks.length === 0) {
      return;
    }

    // Extract key landmark positions
    const positions = {};
    for (const [name, index] of Object.entries(TRACKED_LANDMARKS)) {
      if (landmarks[index]) {
        positions[name] = {
          x: landmarks[index].x,
          y: landmarks[index].y,
          z: landmarks[index].z || 0,
        };
      }
    }

    this.frameHistory.push({
      timestamp: performance.now(),
      positions,
    });

    // Keep only recent frames
    if (this.frameHistory.length > CONFIG.FRAME_HISTORY) {
      this.frameHistory.shift();
    }
  }

  /**
   * Reset the motion history
   */
  reset() {
    this.frameHistory = [];
    this.lastAnalysis = null;
  }

  /**
   * Analyze motion patterns to detect spoofing
   * @returns {Object} Analysis result with score and details
   */
  analyze() {
    if (this.frameHistory.length < CONFIG.MIN_FRAMES_FOR_ANALYSIS) {
      return {
        ready: false,
        framesCollected: this.frameHistory.length,
        framesNeeded: CONFIG.MIN_FRAMES_FOR_ANALYSIS,
      };
    }

    const analysis = {
      ready: true,
      microMovement: this._analyzeMicroMovement(),
      parallelMotion: this._analyzeParallelMotion(),
      motionVariance: this._analyzeMotionVariance(),
    };

    // Calculate overall liveness score from motion
    // Higher score = more likely to be real
    let score = 0.5; // Neutral start

    // Micro-movement check: real faces have natural small movements
    if (analysis.microMovement.avgMovement < CONFIG.STILLNESS_THRESHOLD) {
      // Suspiciously still - likely a photo
      score -= 0.3;
      analysis.flags = analysis.flags || [];
      analysis.flags.push('suspiciously_still');
    } else if (analysis.microMovement.avgMovement > CONFIG.MICRO_MOVEMENT_THRESHOLD) {
      // Has expected natural movement
      score += 0.2;
    }

    // Parallel motion check: if all points move together, likely a phone screen
    if (analysis.parallelMotion.correlation > CONFIG.PARALLEL_MOTION_THRESHOLD) {
      score -= 0.25;
      analysis.flags = analysis.flags || [];
      analysis.flags.push('parallel_motion_detected');
    } else if (analysis.parallelMotion.correlation < 0.5) {
      // Natural independent movement of facial features
      score += 0.15;
    }

    // Motion variance check: real faces have varied motion patterns
    if (analysis.motionVariance.isNatural) {
      score += 0.15;
    } else {
      score -= 0.1;
    }

    analysis.livenessScore = Math.max(0, Math.min(1, score));
    analysis.isLikelyReal = score > 0.5;

    this.lastAnalysis = analysis;
    return analysis;
  }

  /**
   * Analyze micro-movements (involuntary natural face movement)
   */
  _analyzeMicroMovement() {
    const movements = [];

    for (let i = 1; i < this.frameHistory.length; i++) {
      const prev = this.frameHistory[i - 1].positions;
      const curr = this.frameHistory[i].positions;

      let frameMovement = 0;
      let pointCount = 0;

      for (const name of Object.keys(TRACKED_LANDMARKS)) {
        if (prev[name] && curr[name]) {
          const dx = curr[name].x - prev[name].x;
          const dy = curr[name].y - prev[name].y;
          const distance = Math.sqrt(dx * dx + dy * dy);
          frameMovement += distance;
          pointCount++;
        }
      }

      if (pointCount > 0) {
        movements.push(frameMovement / pointCount);
      }
    }

    const avgMovement = movements.length > 0
      ? movements.reduce((a, b) => a + b, 0) / movements.length
      : 0;

    const maxMovement = movements.length > 0 ? Math.max(...movements) : 0;
    const minMovement = movements.length > 0 ? Math.min(...movements) : 0;

    return {
      avgMovement,
      maxMovement,
      minMovement,
      movementRange: maxMovement - minMovement,
    };
  }

  /**
   * Analyze if motion is parallel (all points moving together = phone moving)
   */
  _analyzeParallelMotion() {
    const motionVectors = {};

    // Collect motion vectors for each tracked point
    for (const name of Object.keys(TRACKED_LANDMARKS)) {
      motionVectors[name] = [];
    }

    for (let i = 1; i < this.frameHistory.length; i++) {
      const prev = this.frameHistory[i - 1].positions;
      const curr = this.frameHistory[i].positions;

      for (const name of Object.keys(TRACKED_LANDMARKS)) {
        if (prev[name] && curr[name]) {
          motionVectors[name].push({
            dx: curr[name].x - prev[name].x,
            dy: curr[name].y - prev[name].y,
          });
        }
      }
    }

    // Calculate correlation between motion vectors of different points
    // High correlation = parallel motion = likely a screen/photo
    const correlations = [];
    const names = Object.keys(motionVectors);

    for (let i = 0; i < names.length - 1; i++) {
      for (let j = i + 1; j < names.length; j++) {
        const v1 = motionVectors[names[i]];
        const v2 = motionVectors[names[j]];

        if (v1.length > 0 && v2.length > 0 && v1.length === v2.length) {
          const corr = this._calculateMotionCorrelation(v1, v2);
          if (!isNaN(corr)) {
            correlations.push(corr);
          }
        }
      }
    }

    const avgCorrelation = correlations.length > 0
      ? correlations.reduce((a, b) => a + b, 0) / correlations.length
      : 0;

    return {
      correlation: avgCorrelation,
      pairCount: correlations.length,
    };
  }

  /**
   * Calculate correlation between two motion vector sequences
   */
  _calculateMotionCorrelation(v1, v2) {
    if (v1.length !== v2.length || v1.length === 0) {
      return 0;
    }

    // Calculate correlation for x and y separately, then average
    const n = v1.length;

    const meanX1 = v1.reduce((a, b) => a + b.dx, 0) / n;
    const meanY1 = v1.reduce((a, b) => a + b.dy, 0) / n;
    const meanX2 = v2.reduce((a, b) => a + b.dx, 0) / n;
    const meanY2 = v2.reduce((a, b) => a + b.dy, 0) / n;

    let numeratorX = 0, numeratorY = 0;
    let denomX1 = 0, denomX2 = 0, denomY1 = 0, denomY2 = 0;

    for (let i = 0; i < n; i++) {
      const dx1 = v1[i].dx - meanX1;
      const dy1 = v1[i].dy - meanY1;
      const dx2 = v2[i].dx - meanX2;
      const dy2 = v2[i].dy - meanY2;

      numeratorX += dx1 * dx2;
      numeratorY += dy1 * dy2;
      denomX1 += dx1 * dx1;
      denomX2 += dx2 * dx2;
      denomY1 += dy1 * dy1;
      denomY2 += dy2 * dy2;
    }

    const corrX = denomX1 > 0 && denomX2 > 0
      ? numeratorX / Math.sqrt(denomX1 * denomX2)
      : 0;
    const corrY = denomY1 > 0 && denomY2 > 0
      ? numeratorY / Math.sqrt(denomY1 * denomY2)
      : 0;

    return (Math.abs(corrX) + Math.abs(corrY)) / 2;
  }

  /**
   * Analyze motion variance to check if it's natural
   */
  _analyzeMotionVariance() {
    const variances = [];

    for (const name of Object.keys(TRACKED_LANDMARKS)) {
      const positions = this.frameHistory
        .map(f => f.positions[name])
        .filter(p => p);

      if (positions.length > 5) {
        const xVariance = this._calculateVariance(positions.map(p => p.x));
        const yVariance = this._calculateVariance(positions.map(p => p.y));
        variances.push({ name, xVariance, yVariance });
      }
    }

    // Natural motion: different parts of face have different variance
    // Nose/forehead are more stable, mouth/eyes vary more during expressions
    const varianceValues = variances.map(v => v.xVariance + v.yVariance);
    const varianceSpread = varianceValues.length > 0
      ? Math.max(...varianceValues) - Math.min(...varianceValues)
      : 0;

    return {
      variances,
      varianceSpread,
      isNatural: varianceSpread > 0.00001, // Some variation between facial regions
    };
  }

  /**
   * Calculate variance of an array of numbers
   */
  _calculateVariance(values) {
    if (values.length === 0) return 0;
    const mean = values.reduce((a, b) => a + b, 0) / values.length;
    const squaredDiffs = values.map(v => (v - mean) ** 2);
    return squaredDiffs.reduce((a, b) => a + b, 0) / values.length;
  }

  /**
   * Get a summary suitable for sending to backend
   */
  getSummary() {
    if (!this.lastAnalysis || !this.lastAnalysis.ready) {
      return null;
    }

    return {
      framesAnalyzed: this.frameHistory.length,
      microMovementAvg: this.lastAnalysis.microMovement.avgMovement,
      parallelMotionCorr: this.lastAnalysis.parallelMotion.correlation,
      motionVarianceSpread: this.lastAnalysis.motionVariance.varianceSpread,
      livenessScore: this.lastAnalysis.livenessScore,
      flags: this.lastAnalysis.flags || [],
    };
  }
}

// Singleton instance
let _motionAnalyzer = null;

export function getMotionAnalyzer() {
  if (!_motionAnalyzer) {
    _motionAnalyzer = new MotionAnalyzer();
  }
  return _motionAnalyzer;
}

export function resetMotionAnalyzer() {
  if (_motionAnalyzer) {
    _motionAnalyzer.reset();
  }
}
