"""
Layer 5: Risk Scoring System

Calculates a comprehensive fraud risk score (0-100) based on all verification layers:
- Layer 1: Face Detection quality
- Layer 2: Liveness/Spoof detection
- Layer 3: Deepfake detection
- Layer 4: Duplicate detection
- Additional factors: Behavioral and device signals

Risk Levels:
- 0-30: LOW RISK -> Auto Approve
- 31-60: MEDIUM RISK -> Manual Review
- 61-100: HIGH RISK -> Auto Reject
"""

import logging
from typing import Dict, Optional, Tuple
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)


class RiskLevel(str, Enum):
    LOW = "low"           # 0-30: Auto approve
    MEDIUM = "medium"     # 31-60: Manual review
    HIGH = "high"         # 61-100: Auto reject


class RiskDecision(str, Enum):
    AUTO_APPROVE = "auto_approve"
    MANUAL_REVIEW = "manual_review"
    AUTO_REJECT = "auto_reject"


@dataclass
class RiskAssessment:
    """Complete risk assessment result."""
    score: int                          # 0-100
    level: RiskLevel                    # low/medium/high
    decision: RiskDecision              # auto_approve/manual_review/auto_reject
    factors: Dict[str, float]           # Individual factor scores
    breakdown: Dict[str, Dict]          # Detailed breakdown per layer
    flags: list                         # Risk flags triggered


class RiskScorer:
    """
    Multi-factor risk scoring engine for KYC verification.

    Combines signals from all verification layers into a single
    risk score with actionable decision.
    """

    # Factor weights (must sum to 1.0)
    WEIGHTS = {
        'face_quality': 0.10,      # Layer 1: Face detection quality
        'liveness': 0.25,          # Layer 2: Liveness/spoof score
        'deepfake': 0.25,          # Layer 3: Deepfake detection
        'duplicate': 0.20,         # Layer 4: Duplicate similarity
        'behavioral': 0.10,        # Motion/challenge completion
        'device': 0.10,            # Device/session signals
    }

    # Risk level boundaries
    LOW_RISK_MAX = 30
    MEDIUM_RISK_MAX = 60

    def __init__(self):
        logger.info("RiskScorer initialized")

    def calculate_risk(
        self,
        face_quality: Optional[Dict] = None,
        liveness_result: Optional[Dict] = None,
        deepfake_result: Optional[Dict] = None,
        duplicate_result: Optional[Dict] = None,
        behavioral_data: Optional[Dict] = None,
        device_data: Optional[Dict] = None,
    ) -> RiskAssessment:
        """
        Calculate comprehensive risk score from all verification layers.

        Returns:
            RiskAssessment with score, level, decision, and breakdown
        """
        factors = {}
        breakdown = {}
        flags = []

        # Layer 1: Face Quality Risk
        face_risk, face_breakdown = self._assess_face_quality(face_quality)
        factors['face_quality'] = face_risk
        breakdown['face_quality'] = face_breakdown
        if face_risk > 50:
            flags.append('poor_face_quality')

        # Layer 2: Liveness/Spoof Risk
        liveness_risk, liveness_breakdown = self._assess_liveness(liveness_result)
        factors['liveness'] = liveness_risk
        breakdown['liveness'] = liveness_breakdown
        if liveness_risk > 60:
            flags.append('spoof_suspected')

        # Layer 3: Deepfake Risk
        deepfake_risk, deepfake_breakdown = self._assess_deepfake(deepfake_result)
        factors['deepfake'] = deepfake_risk
        breakdown['deepfake'] = deepfake_breakdown
        if deepfake_risk > 60:
            flags.append('deepfake_suspected')

        # Layer 4: Duplicate Risk
        duplicate_risk, duplicate_breakdown = self._assess_duplicate(duplicate_result)
        factors['duplicate'] = duplicate_risk
        breakdown['duplicate'] = duplicate_breakdown
        if duplicate_risk > 50:
            flags.append('potential_duplicate')

        # Behavioral Risk
        behavioral_risk, behavioral_breakdown = self._assess_behavioral(behavioral_data)
        factors['behavioral'] = behavioral_risk
        breakdown['behavioral'] = behavioral_breakdown
        if behavioral_risk > 50:
            flags.append('suspicious_behavior')

        # Device Risk
        device_risk, device_breakdown = self._assess_device(device_data)
        factors['device'] = device_risk
        breakdown['device'] = device_breakdown
        if device_risk > 50:
            flags.append('suspicious_device')

        # Calculate weighted risk score (0-100)
        weighted_score = sum(
            factors[factor] * weight
            for factor, weight in self.WEIGHTS.items()
        )

        logger.info(f"[RISK] Factor risks (0=safe, 100=risky):")
        for factor, weight in self.WEIGHTS.items():
            risk_val = factors[factor]
            contribution = risk_val * weight
            logger.info(f"  {factor:15s}: risk={risk_val:6.2f} × weight={weight:.2f} = {contribution:6.2f}")
        logger.info(f"[RISK] Weighted sum (before overrides): {weighted_score:.2f}")

        # Apply critical flags that override the score
        if 'deepfake_suspected' in flags and deepfake_risk > 70:
            pre = weighted_score
            weighted_score = max(weighted_score, 65)  # Force high risk
            logger.warning(f"[RISK] Override: deepfake_suspected + risk={deepfake_risk:.1f} → score {pre:.2f} → {weighted_score:.2f}")

        if 'spoof_suspected' in flags and liveness_risk > 70:
            pre = weighted_score
            weighted_score = max(weighted_score, 65)  # Force high risk
            logger.warning(f"[RISK] Override: spoof_suspected + risk={liveness_risk:.1f} → score {pre:.2f} → {weighted_score:.2f}")

        # Round to integer
        final_score = int(round(weighted_score))
        final_score = max(0, min(100, final_score))

        # Determine risk level and decision
        if final_score <= self.LOW_RISK_MAX:
            level = RiskLevel.LOW
            decision = RiskDecision.AUTO_APPROVE
        elif final_score <= self.MEDIUM_RISK_MAX:
            level = RiskLevel.MEDIUM
            decision = RiskDecision.MANUAL_REVIEW
        else:
            level = RiskLevel.HIGH
            decision = RiskDecision.AUTO_REJECT

        logger.info(f"[RISK] RESULT: score={final_score}, level={level.value}, decision={decision.value}, flags={flags}")

        return RiskAssessment(
            score=final_score,
            level=level,
            decision=decision,
            factors=factors,
            breakdown=breakdown,
            flags=flags,
        )

    def _assess_face_quality(self, face_data: Optional[Dict]) -> Tuple[float, Dict]:
        """Convert face detection quality to risk score (0=good, 100=bad)."""
        if not face_data:
            return 50.0, {'status': 'no_data', 'risk': 50}

        risk = 0.0
        details = {}

        confidence = face_data.get('confidence', 0.5)
        details['confidence'] = confidence
        if confidence < 0.7:
            risk += 30
        elif confidence < 0.9:
            risk += 10

        num_faces = face_data.get('num_faces', 1)
        details['num_faces'] = num_faces
        if num_faces > 1:
            risk += 25
        elif num_faces == 0:
            risk = 100

        risk = min(100, risk)
        details['risk'] = risk
        return risk, details

    def _assess_liveness(self, liveness_data: Optional[Dict]) -> Tuple[float, Dict]:
        """Convert liveness/spoof detection to risk score."""
        if not liveness_data:
            return 50.0, {'status': 'no_data', 'risk': 50}

        details = {}

        # Main spoof score (0-1, higher = more real)
        spoof_score = liveness_data.get('score', 0.5)
        details['spoof_score'] = spoof_score

        # Convert to risk (inverse: high authenticity = low risk)
        base_risk = (1 - spoof_score) * 100

        # Motion analysis integration
        motion = liveness_data.get('motion_analysis', {})
        if motion:
            liveness_score = motion.get('liveness_score') or 0.5
            details['motion_liveness'] = liveness_score
            if liveness_score < 0.4:
                base_risk += 10

        risk = min(100, base_risk)
        details['risk'] = risk
        return risk, details

    def _assess_deepfake(self, deepfake_data: Optional[Dict]) -> Tuple[float, Dict]:
        """Convert deepfake detection to risk score."""
        if not deepfake_data:
            return 40.0, {'status': 'no_data', 'risk': 40}

        details = {}

        authenticity = deepfake_data.get('score', 0.5)
        details['authenticity_score'] = authenticity

        # Convert to risk
        base_risk = (1 - authenticity) * 100

        critical_flag = deepfake_data.get('critical_flag')
        if critical_flag:
            base_risk += 20
            details['critical_flag'] = critical_flag

        risk = min(100, base_risk)
        details['risk'] = risk
        return risk, details

    def _assess_duplicate(self, duplicate_data: Optional[Dict]) -> Tuple[float, Dict]:
        """Convert duplicate detection to risk score."""
        if not duplicate_data:
            return 0.0, {'status': 'no_duplicates', 'risk': 0}

        details = {}
        matches = duplicate_data.get('matches', [])
        details['num_matches'] = len(matches)

        if matches:
            top_match = matches[0]
            similarity = top_match.get('similarity', 0)
            details['top_similarity'] = similarity

            if similarity > 0.9:
                risk = 90
            elif similarity > 0.8:
                risk = 70
            elif similarity > 0.7:
                risk = 50
            elif similarity > 0.6:
                risk = 30
            else:
                risk = 10
        else:
            risk = 0

        details['risk'] = risk
        return risk, details

    def _assess_behavioral(self, behavioral_data: Optional[Dict]) -> Tuple[float, Dict]:
        """Assess behavioral signals for risk."""
        if not behavioral_data:
            return 30.0, {'status': 'no_data', 'risk': 30}

        risk = 20.0
        details = {}

        challenges_completed = behavioral_data.get('challenges_completed', [])
        num_challenges = len(challenges_completed)
        details['challenges_completed'] = num_challenges

        if num_challenges >= 3:
            risk -= 10
        elif num_challenges < 2:
            risk += 20

        motion = behavioral_data.get('motion_analysis', {})
        if motion:
            liveness_score = motion.get('liveness_score') or 0.5
            details['motion_liveness'] = liveness_score

            if liveness_score > 0.7:
                risk -= 10
            elif liveness_score < 0.3:
                risk += 20

        risk = max(0, min(100, risk))
        details['risk'] = risk
        return risk, details

    def _assess_device(self, device_data: Optional[Dict]) -> Tuple[float, Dict]:
        """Assess device/session signals for risk."""
        if not device_data:
            return 25.0, {'status': 'no_data', 'risk': 25}

        risk = 15.0
        details = {}

        user_agent = device_data.get('user_agent') or ''
        suspicious_ua = ['headless', 'phantom', 'selenium', 'puppeteer', 'bot']
        if any(s in user_agent.lower() for s in suspicious_ua):
            risk += 40
            details['suspicious_ua'] = True

        session_attempts = device_data.get('session_attempts', 1)
        details['session_attempts'] = session_attempts
        if session_attempts > 5:
            risk += 20
        elif session_attempts > 3:
            risk += 10

        risk = max(0, min(100, risk))
        details['risk'] = risk
        return risk, details


# Global instance
_scorer = None


def get_risk_scorer() -> RiskScorer:
    """Get or create the global risk scorer instance."""
    global _scorer
    if _scorer is None:
        _scorer = RiskScorer()
    return _scorer


def calculate_risk_score(
    face_quality: Optional[Dict] = None,
    liveness_result: Optional[Dict] = None,
    deepfake_result: Optional[Dict] = None,
    duplicate_result: Optional[Dict] = None,
    behavioral_data: Optional[Dict] = None,
    device_data: Optional[Dict] = None,
) -> RiskAssessment:
    """
    Calculate comprehensive risk score.

    Returns RiskAssessment with:
    - score: 0-100 (higher = more risky)
    - level: low/medium/high
    - decision: auto_approve/manual_review/auto_reject
    """
    scorer = get_risk_scorer()
    return scorer.calculate_risk(
        face_quality=face_quality,
        liveness_result=liveness_result,
        deepfake_result=deepfake_result,
        duplicate_result=duplicate_result,
        behavioral_data=behavioral_data,
        device_data=device_data,
    )
