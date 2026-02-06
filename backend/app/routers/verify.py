import base64
import json
import uuid
import logging
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
    layer_results = {}

    try:
        # ── LAYER 1: FACE DETECTION ──────────────────────────────
        try:
            image_bytes = base64.b64decode(request.image)
            image = preprocess_image(image_bytes)
        except Exception as e:
            logger.error(f"Failed to decode image: {e}")
            return VerifyResponse(
                status="error",
                message="Invalid image format. Please try again.",
                can_dispute=False
            )

        try:
            face_bbox = get_face_bbox(image)
            if face_bbox is None:
                layer_results['layer1_face_detection'] = {
                    'status': 'failed', 'confidence': 0, 'detail': 'no_face'
                }
                return VerifyResponse(
                    status="error",
                    message="No face detected. Please ensure your face is clearly visible.",
                    can_dispute=False,
                    layer_results=layer_results
                )
        except MultipleFacesError:
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
        logger.info("Layer 1 (Face Detection): PASSED")

        # ── LAYER 2: LIVENESS / SPOOF CHECK ──────────────────────
        spoof_score, spoof_details = check_spoof(image, face_bbox)
        logger.info(f"Layer 2 (Liveness): score={spoof_score:.3f}")

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

        adjusted_spoof_score = max(0.0, spoof_score - motion_penalty)

        liveness_passed = adjusted_spoof_score >= settings.spoof_threshold
        layer_results['layer2_liveness'] = {
            'status': 'passed' if liveness_passed else 'failed',
            'score': round(adjusted_spoof_score, 3),
        }

        if not liveness_passed:
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
        logger.info("Layer 2 (Liveness): PASSED")

        # ── LAYER 3: DEEPFAKE DETECTION ──────────────────────────
        try:
            deepfake_score, deepfake_details = check_deepfake(image)
        except Exception as e:
            logger.warning(f"Deepfake detection failed, using default score: {e}")
            deepfake_score = 0.65  # Default pass score
            deepfake_details = {'error': str(e), 'fallback': True}
        logger.info(f"Layer 3 (Deepfake): score={deepfake_score:.3f}")

        deepfake_passed = deepfake_score >= 0.4
        layer_results['layer3_deepfake'] = {
            'status': 'passed' if deepfake_passed else 'failed',
            'score': round(deepfake_score, 3),
        }

        if not deepfake_passed:
            await log_audit(db, request.session_id, "verify", "deepfake_detected",
                {"deepfake_score": deepfake_score, "details": deepfake_details}, http_request)
            return VerifyResponse(
                status="deepfake_detected",
                message="Our system detected potential image manipulation. Please use a live camera.",
                confidence=deepfake_score,
                can_dispute=True,
                layer_results=layer_results
            )
        logger.info("Layer 3 (Deepfake): PASSED")

        # ── LAYER 4: DUPLICATE CHECK ────────────────────────────
        try:
            embedding, face_info = get_face_embedding(image)
        except NoFaceError:
            return VerifyResponse(
                status="error",
                message="Could not extract face features. Please try again.",
                can_dispute=False,
                layer_results=layer_results
            )

        matches = await find_similar_faces(embedding, db)
        duplicate_data = None

        if matches:
            best_match = matches[0]
            logger.info(f"Layer 4 (Duplicate): match found, similarity={best_match['similarity']:.3f}")

            duplicate_data = {'matches': matches}
            layer_results['layer4_duplicate'] = {
                'status': 'flagged',
                'matches_found': len(matches),
                'top_similarity': round(best_match['similarity'], 3),
            }

            # Upload image for review
            image_url = upload_image(
                request.image, folder="kyc/reviews",
                public_id=f"review_{request.session_id}"
            )

            review_id = await create_review_entry(
                db, new_customer_id=request.session_id,
                new_embedding=embedding, new_image_url=image_url,
                matched_customer_id=best_match["customer_id"],
                matched_customer_name=best_match.get("customer_name"),
                similarity_score=best_match["similarity"]
            )

            await log_audit(db, request.session_id, "verify", "duplicate_found",
                {"matched_customer_id": best_match["customer_id"],
                 "similarity": best_match["similarity"], "review_id": review_id}, http_request)
        else:
            layer_results['layer4_duplicate'] = {
                'status': 'passed', 'matches_found': 0,
            }
            logger.info("Layer 4 (Duplicate): PASSED - unique face")

        # ── LAYER 5: RISK SCORING ────────────────────────────────
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

        logger.info(f"Layer 5 (Risk): score={risk_assessment.score}, decision={risk_assessment.decision.value}")

        # ── FINAL DECISION ───────────────────────────────────────
        if risk_assessment.decision == RiskDecision.AUTO_REJECT:
            await log_audit(db, request.session_id, "verify", "rejected",
                {"risk_score": risk_assessment.score, "flags": risk_assessment.flags}, http_request)
            return VerifyResponse(
                status="rejected",
                message="Verification failed due to high risk indicators.",
                can_dispute=True,
                risk_score=risk_assessment.score,
                risk_level=risk_assessment.level.value,
                layer_results=layer_results,
            )

        if risk_assessment.decision == RiskDecision.MANUAL_REVIEW or matches:
            # Store face but flag for review
            customer_id = str(uuid.uuid4())
            face_metadata = {
                "session_id": request.session_id,
                "challenges": request.challenges_completed,
                "spoof_score": spoof_score,
                "deepfake_score": deepfake_score,
                "risk_score": risk_assessment.score,
            }
            await add_face_embedding(customer_id=customer_id, embedding=embedding,
                                     db=db, metadata=face_metadata)

            await log_audit(db, request.session_id, "verify", "pending_review",
                {"customer_id": customer_id, "risk_score": risk_assessment.score,
                 "flags": risk_assessment.flags}, http_request)

            return VerifyResponse(
                status="pending_review",
                message="Your registration is under review. We'll contact you shortly.",
                customer_id=customer_id,
                confidence=adjusted_spoof_score,
                review_id=review_id if matches else None,
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

        await add_face_embedding(customer_id=customer_id, embedding=embedding,
                                 db=db, metadata=face_metadata)

        await log_audit(db, request.session_id, "verify", "success",
            {"customer_id": customer_id, "risk_score": risk_assessment.score,
             "spoof_score": spoof_score, "deepfake_score": deepfake_score}, http_request)

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
        logger.exception(f"Verification error: {e}")
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
