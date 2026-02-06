import base64
import json
import time
import uuid
import logging
import numpy as np
from fastapi import APIRouter, Depends, HTTPException, Request
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import text

from app.db.session import get_db
from app.config import get_settings
from app.models.schemas import (
    VerifyRequest,
    VerifyResponse,
    DisputeRequest,
    DisputeResponse
)
from app.services.face_embedder import (
    get_face_embedding,
    get_face_bbox,
    preprocess_image,
    NoFaceError,
    MultipleFacesError
)
from app.services.spoof_detector import check_spoof
from app.services.deepfake_detector import check_deepfake
from app.services.deduplication import find_similar_faces, add_face_embedding
from app.services.storage import upload_image
from app.services.risk_scorer import calculate_risk_score, RiskDecision

logger = logging.getLogger(__name__)
settings = get_settings()

router = APIRouter(prefix="/api", tags=["verification"])


def _log_separator(label: str):
    logger.info(f"{'='*60}")
    logger.info(f"  {label}")
    logger.info(f"{'='*60}")


@router.post("/verify", response_model=VerifyResponse)
async def verify_face(
    request: VerifyRequest,
    http_request: Request,
    db: AsyncSession = Depends(get_db)
):
    """
    5-Layer Verification Pipeline:
    Layer 1: Face Detection
    Layer 2: Liveness/Spoof Check
    Layer 3: Deepfake Detection
    Layer 4: Duplicate Check
    Layer 5: Risk Scoring (0-100)
    """
    pipeline_start = time.time()
    layer_results = {}

    _log_separator("VERIFICATION PIPELINE START")
    logger.info(f"Session ID      : {request.session_id}")
    logger.info(f"Challenges done : {request.challenges_completed}")
    logger.info(f"Has motion data : {request.motion_analysis is not None}")
    if request.motion_analysis:
        ma = request.motion_analysis
        logger.info(f"  Motion frames_analyzed      : {ma.frames_analyzed}")
        logger.info(f"  Motion micro_movement_avg   : {ma.micro_movement_avg}")
        logger.info(f"  Motion parallel_motion_corr : {ma.parallel_motion_corr}")
        logger.info(f"  Motion motion_variance      : {ma.motion_variance_spread}")
        logger.info(f"  Motion liveness_score        : {ma.liveness_score}")
        logger.info(f"  Motion flags                 : {ma.flags}")
    logger.info(f"Image size (b64): {len(request.image)} chars")
    logger.info(f"Client IP       : {http_request.client.host if http_request.client else 'unknown'}")
    logger.info(f"User-Agent      : {http_request.headers.get('user-agent', 'none')}")

    try:
        # ── IDEMPOTENCY GUARD: Prevent duplicate processing ──────
        logger.info("[IDEMPOTENCY] Checking if session already processed...")
        existing = await db.execute(
            text("""
                SELECT result, details FROM audit_log
                WHERE session_id = :session_id AND action = 'verify'
                ORDER BY created_at DESC LIMIT 1
            """),
            {"session_id": request.session_id}
        )
        existing_row = existing.fetchone()
        if existing_row and existing_row.result in ('success', 'pending_review', 'duplicate_found'):
            logger.info(f"[IDEMPOTENCY] BLOCKED — session already processed as '{existing_row.result}'")
            return VerifyResponse(
                status=existing_row.result if existing_row.result != 'duplicate_found' else 'pending_review',
                message="This session has already been processed.",
                can_dispute=False,
            )
        logger.info(f"[IDEMPOTENCY] OK — no prior result (existing={existing_row.result if existing_row else 'none'})")

        # ── LAYER 1: FACE DETECTION ──────────────────────────────
        _log_separator("LAYER 1: FACE DETECTION")
        t1 = time.time()

        try:
            image_bytes = base64.b64decode(request.image)
            logger.info(f"[L1] Decoded base64 → {len(image_bytes)} raw bytes")
            image = preprocess_image(image_bytes)
            logger.info(f"[L1] Image decoded → shape={image.shape}, dtype={image.dtype}")
        except Exception as e:
            logger.error(f"[L1] FAILED — image decode error: {type(e).__name__}: {e}")
            return VerifyResponse(
                status="error",
                message="Invalid image format. Please try again.",
                can_dispute=False
            )

        try:
            face_bbox = get_face_bbox(image)
            if face_bbox is None:
                logger.warning(f"[L1] FAILED — no face detected in {image.shape} image")
                layer_results['layer1_face_detection'] = {
                    'status': 'failed', 'confidence': 0, 'detail': 'no_face'
                }
                return VerifyResponse(
                    status="error",
                    message="No face detected. Please ensure your face is clearly visible.",
                    can_dispute=False,
                    layer_results=layer_results
                )
        except MultipleFacesError as e:
            logger.warning(f"[L1] FAILED — multiple faces: {e}")
            layer_results['layer1_face_detection'] = {
                'status': 'failed', 'confidence': 0, 'detail': 'multiple_faces'
            }
            return VerifyResponse(
                status="error",
                message="Multiple faces detected. Please ensure only you are in the frame.",
                can_dispute=False,
                layer_results=layer_results
            )

        layer_results['layer1_face_detection'] = {
            'status': 'passed',
            'confidence': 0.95,
        }
        face_quality_data = {'confidence': 0.95, 'num_faces': 1, 'bbox': face_bbox}
        logger.info(f"[L1] PASSED — bbox={face_bbox}, time={time.time()-t1:.3f}s")

        # ── LAYER 2: LIVENESS / SPOOF CHECK ──────────────────────
        _log_separator("LAYER 2: LIVENESS / SPOOF CHECK")
        t2 = time.time()

        spoof_score, spoof_details = check_spoof(image, face_bbox)
        logger.info(f"[L2] Raw spoof score: {spoof_score:.4f} (threshold={settings.spoof_threshold})")
        logger.info(f"[L2] Spoof sub-scores:")
        for key, val in spoof_details.get('individual_scores', {}).items():
            logger.info(f"  {key:12s}: {val:.4f}" if isinstance(val, float) else f"  {key:12s}: {val}")
        if spoof_details.get('critical_failures'):
            logger.warning(f"[L2] Critical failures: {spoof_details['critical_failures']}")
        if spoof_details.get('dl_score') is not None:
            logger.info(f"[L2] DL model score: {spoof_details['dl_score']:.4f}")
        else:
            logger.info("[L2] DL model: NOT LOADED (multi-technique fallback)")

        # Integrate client-side motion analysis
        motion_penalty = 0.0
        motion_flags = []
        if request.motion_analysis:
            ma = request.motion_analysis
            if ma.liveness_score is not None and ma.liveness_score < 0.4:
                motion_penalty = 0.15
                motion_flags.append("low_motion_liveness")
            if ma.flags:
                if "suspiciously_still" in ma.flags:
                    motion_penalty += 0.1
                    motion_flags.append("static_face_detected")
                if "parallel_motion_detected" in ma.flags:
                    motion_penalty += 0.1
                    motion_flags.append("screen_motion_pattern")
            spoof_details["motion_analysis"] = {
                "liveness_score": ma.liveness_score,
                "flags": ma.flags,
                "penalty_applied": motion_penalty
            }
            logger.info(f"[L2] Motion penalty: {motion_penalty:.2f}, flags={motion_flags}")
        else:
            logger.info("[L2] No client motion analysis data received")

        adjusted_spoof_score = max(0.0, spoof_score - motion_penalty)
        logger.info(f"[L2] Adjusted score: {adjusted_spoof_score:.4f} (raw={spoof_score:.4f} - penalty={motion_penalty:.2f})")

        liveness_passed = adjusted_spoof_score >= settings.spoof_threshold
        layer_results['layer2_liveness'] = {
            'status': 'passed' if liveness_passed else 'failed',
            'score': round(adjusted_spoof_score, 3),
        }

        if not liveness_passed:
            logger.warning(f"[L2] FAILED — {adjusted_spoof_score:.4f} < threshold {settings.spoof_threshold}")
            await log_audit(db, request.session_id, "verify", "spoof_detected",
                {"spoof_score": spoof_score, "adjusted_score": adjusted_spoof_score,
                 "motion_flags": motion_flags}, http_request)
            return VerifyResponse(
                status="spoof_detected",
                message="We couldn't verify that you're physically present. Please use a live camera.",
                confidence=adjusted_spoof_score,
                can_dispute=True,
                layer_results=layer_results
            )
        logger.info(f"[L2] PASSED — time={time.time()-t2:.3f}s")

        # ── LAYER 3: DEEPFAKE DETECTION ──────────────────────────
        _log_separator("LAYER 3: DEEPFAKE DETECTION")
        t3 = time.time()

        try:
            deepfake_score, deepfake_details = check_deepfake(image, face_bbox=face_bbox)
        except Exception as e:
            logger.warning(f"[L3] Detection exception, using fallback: {type(e).__name__}: {e}")
            deepfake_score = 0.65
            deepfake_details = {'error': str(e), 'fallback': True}

        logger.info(f"[L3] Authenticity score: {deepfake_score:.4f} (pass threshold=0.4)")
        logger.info(f"[L3] Method: {deepfake_details.get('method', 'unknown')}")
        if 'ml_model' in deepfake_details:
            ml = deepfake_details['ml_model']
            logger.info(f"[L3] ML model output: real_prob={ml.get('real_probability')}, "
                       f"deepfake_prob={ml.get('deepfake_probability')}, "
                       f"prediction={ml.get('prediction')}, confidence={ml.get('confidence')}")
            if ml.get('error'):
                logger.warning(f"[L3] ML model error: {ml['error']}")
        if 'frequency_analysis' in deepfake_details:
            freq = deepfake_details['frequency_analysis']
            logger.info(f"[L3] Frequency analysis: mid_energy={freq.get('mid_frequency_energy')}, "
                       f"high_energy={freq.get('high_frequency_energy')}, "
                       f"smoothness={freq.get('profile_smoothness')}, score={freq.get('score')}")
        if deepfake_details.get('critical_flag'):
            logger.warning(f"[L3] CRITICAL FLAG: {deepfake_details['critical_flag']}")

        deepfake_passed = deepfake_score >= 0.4
        layer_results['layer3_deepfake'] = {
            'status': 'passed' if deepfake_passed else 'failed',
            'score': round(deepfake_score, 3),
        }

        if not deepfake_passed:
            logger.warning(f"[L3] FAILED — {deepfake_score:.4f} < 0.4")
            await log_audit(db, request.session_id, "verify", "deepfake_detected",
                {"deepfake_score": deepfake_score, "details": deepfake_details}, http_request)
            return VerifyResponse(
                status="deepfake_detected",
                message="Our system detected potential image manipulation. Please use a live camera.",
                confidence=deepfake_score,
                can_dispute=True,
                layer_results=layer_results
            )
        logger.info(f"[L3] PASSED — time={time.time()-t3:.3f}s")

        # ── LAYER 4: DUPLICATE CHECK ────────────────────────────
        _log_separator("LAYER 4: DUPLICATE CHECK")
        t4 = time.time()

        try:
            embedding, face_info = get_face_embedding(image)
            logger.info(f"[L4] Embedding extracted: dim={embedding.shape}, norm={float(np.linalg.norm(embedding)):.4f}")
            logger.info(f"[L4] Face info: det_score={face_info.get('det_score')}, bbox={face_info.get('bbox')}")
        except NoFaceError:
            logger.error("[L4] FAILED — no face for embedding extraction")
            return VerifyResponse(
                status="error",
                message="Could not extract face features. Please try again.",
                can_dispute=False,
                layer_results=layer_results
            )

        logger.info(f"[L4] Searching DB for duplicates (threshold={settings.duplicate_threshold})...")
        matches = await find_similar_faces(embedding, db)
        duplicate_data = None

        if matches:
            best_match = matches[0]
            logger.warning(f"[L4] FLAGGED — {len(matches)} match(es) found:")
            for i, m in enumerate(matches):
                logger.warning(f"  Match {i+1}: customer_id={m['customer_id']}, "
                             f"similarity={m['similarity']:.4f}, distance={m['distance']:.4f}")

            duplicate_data = {'matches': matches}
            layer_results['layer4_duplicate'] = {
                'status': 'flagged',
                'matches_found': len(matches),
                'top_similarity': round(best_match['similarity'], 3),
            }

            # Upload image for review
            logger.info("[L4] Uploading image for review...")
            image_url = upload_image(
                request.image, folder="kyc/reviews",
                public_id=f"review_{request.session_id}"
            )
            logger.info(f"[L4] Image uploaded: {image_url or 'FAILED'}")

            review_id = await create_review_entry(
                db, new_customer_id=request.session_id,
                new_embedding=embedding, new_image_url=image_url,
                matched_customer_id=best_match["customer_id"],
                matched_customer_name=best_match.get("customer_name"),
                similarity_score=best_match["similarity"]
            )
            logger.info(f"[L4] Review entry created: {review_id}")

            await log_audit(db, request.session_id, "verify", "duplicate_found",
                {"matched_customer_id": best_match["customer_id"],
                 "similarity": best_match["similarity"], "review_id": review_id}, http_request)
        else:
            layer_results['layer4_duplicate'] = {
                'status': 'passed', 'matches_found': 0,
            }
            logger.info(f"[L4] PASSED — no duplicates found, time={time.time()-t4:.3f}s")

        # ── LAYER 5: RISK SCORING ────────────────────────────────
        _log_separator("LAYER 5: RISK SCORING")
        t5 = time.time()

        liveness_for_risk = {
            'score': adjusted_spoof_score,
            'motion_analysis': spoof_details.get('motion_analysis', {}),
        }
        deepfake_for_risk = {
            'score': deepfake_score,
            'details': deepfake_details,
            'critical_flag': deepfake_details.get('critical_flag'),
        }
        behavioral_for_risk = {
            'challenges_completed': request.challenges_completed,
            'motion_analysis': {
                'liveness_score': request.motion_analysis.liveness_score if request.motion_analysis else None,
            },
        }
        device_for_risk = {
            'user_agent': http_request.headers.get("user-agent", ""),
        }

        logger.info("[L5] Risk scorer inputs:")
        logger.info(f"  face_quality  : {face_quality_data}")
        logger.info(f"  liveness      : {liveness_for_risk}")
        logger.info(f"  deepfake      : score={deepfake_score:.4f}, critical={deepfake_details.get('critical_flag')}")
        logger.info(f"  duplicate     : {'matches found' if duplicate_data else 'no matches'}")
        logger.info(f"  behavioral    : challenges={request.challenges_completed}")
        logger.info(f"  device        : ua={http_request.headers.get('user-agent', '')[:80]}")

        risk_assessment = calculate_risk_score(
            face_quality=face_quality_data,
            liveness_result=liveness_for_risk,
            deepfake_result=deepfake_for_risk,
            duplicate_result=duplicate_data,
            behavioral_data=behavioral_for_risk,
            device_data=device_for_risk,
        )

        layer_results['layer5_risk_score'] = {
            'score': risk_assessment.score,
            'level': risk_assessment.level.value,
            'decision': risk_assessment.decision.value,
            'flags': risk_assessment.flags,
        }

        logger.info(f"[L5] Risk assessment result:")
        logger.info(f"  Final score   : {risk_assessment.score}/100")
        logger.info(f"  Level         : {risk_assessment.level.value}")
        logger.info(f"  Decision      : {risk_assessment.decision.value}")
        logger.info(f"  Flags         : {risk_assessment.flags}")
        logger.info(f"  Factor scores : {risk_assessment.factors}")
        logger.info(f"  Breakdown     :")
        for layer_name, breakdown in risk_assessment.breakdown.items():
            logger.info(f"    {layer_name:15s}: {breakdown}")
        logger.info(f"[L5] time={time.time()-t5:.3f}s")

        # ── FINAL DECISION ───────────────────────────────────────
        _log_separator("FINAL DECISION")

        if risk_assessment.decision == RiskDecision.AUTO_REJECT:
            logger.warning(f"[DECISION] AUTO_REJECT — risk={risk_assessment.score}, flags={risk_assessment.flags}")
            await log_audit(db, request.session_id, "verify", "rejected",
                {"risk_score": risk_assessment.score, "flags": risk_assessment.flags}, http_request)
            elapsed = time.time() - pipeline_start
            logger.info(f"[PIPELINE] Completed in {elapsed:.3f}s → REJECTED")
            return VerifyResponse(
                status="rejected",
                message="Verification failed due to high risk indicators.",
                can_dispute=True,
                risk_score=risk_assessment.score,
                risk_level=risk_assessment.level.value,
                layer_results=layer_results,
            )

        if matches:
            customer_id = request.session_id

            logger.info(f"[DECISION] PENDING_REVIEW (duplicate) — risk={risk_assessment.score}, customer_id={customer_id}")
            await log_audit(db, request.session_id, "verify", "pending_review",
                {"customer_id": customer_id, "risk_score": risk_assessment.score,
                 "flags": risk_assessment.flags}, http_request)

            elapsed = time.time() - pipeline_start
            logger.info(f"[PIPELINE] Completed in {elapsed:.3f}s → PENDING_REVIEW (duplicate)")
            return VerifyResponse(
                status="pending_review",
                message="Your registration is under review. We'll contact you shortly.",
                customer_id=customer_id,
                confidence=adjusted_spoof_score,
                review_id=review_id,
                risk_score=risk_assessment.score,
                risk_level=risk_assessment.level.value,
                layer_results=layer_results,
            )

        if risk_assessment.decision == RiskDecision.MANUAL_REVIEW:
            customer_id = str(uuid.uuid4())
            face_metadata = {
                "session_id": request.session_id,
                "challenges": request.challenges_completed,
                "spoof_score": spoof_score,
                "deepfake_score": deepfake_score,
                "risk_score": risk_assessment.score,
            }
            logger.info(f"[DECISION] PENDING_REVIEW (medium risk) — storing embedding, customer_id={customer_id}")
            await add_face_embedding(customer_id=customer_id, embedding=embedding,
                                     db=db, metadata=face_metadata)
            logger.info(f"[DECISION] Embedding stored in customer_faces")

            await log_audit(db, request.session_id, "verify", "pending_review",
                {"customer_id": customer_id, "risk_score": risk_assessment.score,
                 "flags": risk_assessment.flags}, http_request)

            elapsed = time.time() - pipeline_start
            logger.info(f"[PIPELINE] Completed in {elapsed:.3f}s → PENDING_REVIEW (medium risk)")
            return VerifyResponse(
                status="pending_review",
                message="Your registration is under review. We'll contact you shortly.",
                customer_id=customer_id,
                confidence=adjusted_spoof_score,
                risk_score=risk_assessment.score,
                risk_level=risk_assessment.level.value,
                layer_results=layer_results,
            )

        # AUTO APPROVE - low risk
        customer_id = str(uuid.uuid4())
        face_metadata = {
            "session_id": request.session_id,
            "challenges": request.challenges_completed,
            "spoof_score": spoof_score,
            "deepfake_score": deepfake_score,
            "risk_score": risk_assessment.score,
        }
        if request.motion_analysis:
            face_metadata["motion_liveness_score"] = request.motion_analysis.liveness_score

        logger.info(f"[DECISION] AUTO_APPROVE — risk={risk_assessment.score}, customer_id={customer_id}")
        await add_face_embedding(customer_id=customer_id, embedding=embedding,
                                 db=db, metadata=face_metadata)
        logger.info(f"[DECISION] Embedding stored in customer_faces")

        await log_audit(db, request.session_id, "verify", "success",
            {"customer_id": customer_id, "risk_score": risk_assessment.score,
             "spoof_score": spoof_score, "deepfake_score": deepfake_score}, http_request)

        elapsed = time.time() - pipeline_start
        logger.info(f"[PIPELINE] Completed in {elapsed:.3f}s → SUCCESS (auto approved)")
        return VerifyResponse(
            status="success",
            message="Verification successful. Your identity has been confirmed.",
            customer_id=customer_id,
            confidence=adjusted_spoof_score,
            risk_score=risk_assessment.score,
            risk_level=risk_assessment.level.value,
            layer_results=layer_results,
        )

    except Exception as e:
        elapsed = time.time() - pipeline_start
        logger.exception(f"[PIPELINE] EXCEPTION after {elapsed:.3f}s: {type(e).__name__}: {e}")
        return VerifyResponse(
            status="error",
            message=f"Verification error: {type(e).__name__}: {str(e)}",
            can_dispute=True,
            layer_results=layer_results
        )


@router.post("/verify/dispute", response_model=DisputeResponse)
async def submit_dispute(
    request: DisputeRequest,
    http_request: Request,
    db: AsyncSession = Depends(get_db)
):
    """
    Submit a dispute when user claims they are real despite spoof detection.
    """
    try:
        # Upload image for manual review
        image_url = upload_image(
            request.image,
            folder="kyc/disputes",
            public_id=f"dispute_{request.session_id}"
        )

        # Store dispute record
        query = text("""
            INSERT INTO dispute_images (session_id, image_url, reason)
            VALUES (:session_id, :image_url, :reason)
            RETURNING id
        """)

        result = await db.execute(
            query,
            {
                "session_id": request.session_id,
                "image_url": image_url or "upload_failed",
                "reason": request.reason
            }
        )
        await db.commit()

        # Log the dispute
        await log_audit(
            db,
            request.session_id,
            "dispute",
            "submitted",
            {"reason": request.reason},
            http_request
        )

        return DisputeResponse(
            status="submitted",
            message="Your dispute has been submitted. Our team will review it and contact you."
        )

    except Exception as e:
        logger.exception(f"Dispute submission error: {e}")
        raise HTTPException(status_code=500, detail="Failed to submit dispute")


async def create_review_entry(
    db: AsyncSession,
    new_customer_id: str,
    new_embedding,
    new_image_url: str,
    matched_customer_id: str,
    matched_customer_name: str,
    similarity_score: float
) -> str:
    """Create an entry in the review queue"""
    query = text("""
        INSERT INTO review_queue (
            new_customer_id,
            new_embedding,
            new_face_image_url,
            matched_customer_id,
            matched_customer_name,
            similarity_score
        )
        VALUES (
            :new_customer_id,
            CAST(:new_embedding AS vector),
            :new_image_url,
            :matched_customer_id,
            :matched_customer_name,
            :similarity_score
        )
        RETURNING id
    """)

    result = await db.execute(
        query,
        {
            "new_customer_id": new_customer_id,
            "new_embedding": '[' + ','.join(map(str, new_embedding.tolist())) + ']',
            "new_image_url": new_image_url,
            "matched_customer_id": matched_customer_id,
            "matched_customer_name": matched_customer_name,
            "similarity_score": similarity_score
        }
    )
    await db.commit()

    row = result.fetchone()
    return str(row.id)


async def log_audit(
    db: AsyncSession,
    session_id: str,
    action: str,
    result: str,
    details: dict,
    request: Request
):
    """Log an audit entry"""
    try:
        # Get client info
        ip_address = request.client.host if request.client else None
        user_agent = request.headers.get("user-agent")

        query = text("""
            INSERT INTO audit_log (session_id, action, result, details, ip_address, user_agent)
            VALUES (:session_id, :action, :result, CAST(:details AS jsonb), :ip_address, :user_agent)
        """)

        await db.execute(
            query,
            {
                "session_id": session_id,
                "action": action,
                "result": result,
                "details": json.dumps(details, default=str),
                "ip_address": ip_address,
                "user_agent": user_agent
            }
        )
        await db.commit()
    except Exception as e:
        logger.error(f"Failed to log audit: {e}")
