import base64
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
from app.services.deduplication import find_similar_faces, add_face_embedding
from app.services.storage import upload_image

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
    Verify a face image for liveness and uniqueness.

    Process:
    1. Decode image
    2. Detect face
    3. Check for spoofing
    4. Extract embedding
    5. Check for duplicates
    6. Store if unique, flag if duplicate
    """
    try:
        # 1. Decode image
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

        # 2. Detect face and get bbox
        try:
            face_bbox = get_face_bbox(image)
            if face_bbox is None:
                return VerifyResponse(
                    status="error",
                    message="No face detected in the image. Please ensure your face is clearly visible.",
                    can_dispute=False
                )
        except MultipleFacesError:
            return VerifyResponse(
                status="error",
                message="Multiple faces detected. Please ensure only you are in the frame.",
                can_dispute=False
            )

        # 3. Check for spoofing (image analysis)
        spoof_score, spoof_details = check_spoof(image, face_bbox)
        logger.info(f"Spoof check: score={spoof_score}, details={spoof_details}")

        # 3b. Validate client-side motion analysis (if provided)
        motion_penalty = 0.0
        motion_flags = []
        if request.motion_analysis:
            ma = request.motion_analysis
            logger.info(f"Motion analysis: liveness_score={ma.liveness_score}, flags={ma.flags}")

            # Check for suspicious motion patterns
            if ma.liveness_score is not None and ma.liveness_score < 0.4:
                motion_penalty = 0.15  # Reduce overall score
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

        # Apply motion penalty to spoof score
        adjusted_spoof_score = max(0.0, spoof_score - motion_penalty)

        if adjusted_spoof_score < settings.spoof_threshold:
            # Log the attempt
            await log_audit(
                db,
                request.session_id,
                "verify",
                "spoof_detected",
                {
                    "spoof_score": spoof_score,
                    "adjusted_score": adjusted_spoof_score,
                    "motion_penalty": motion_penalty,
                    "motion_flags": motion_flags,
                    "spoof_details": spoof_details,
                    "challenges": request.challenges_completed
                },
                http_request
            )

            return VerifyResponse(
                status="spoof_detected",
                message="We couldn't verify that you're physically present. Please ensure you're using a live camera, not a photo or video.",
                confidence=adjusted_spoof_score,
                can_dispute=True
            )

        # 4. Extract face embedding
        try:
            embedding, face_info = get_face_embedding(image)
        except NoFaceError:
            return VerifyResponse(
                status="error",
                message="Could not extract face features. Please try again.",
                can_dispute=False
            )

        # 5. Check for duplicates
        matches = await find_similar_faces(embedding, db)

        if matches:
            best_match = matches[0]
            logger.info(f"Duplicate found: {best_match}")

            # Upload image for review
            image_url = upload_image(
                request.image,
                folder="kyc/reviews",
                public_id=f"review_{request.session_id}"
            )

            # Create review queue entry
            review_id = await create_review_entry(
                db,
                new_customer_id=request.session_id,
                new_embedding=embedding,
                new_image_url=image_url,
                matched_customer_id=best_match["customer_id"],
                matched_customer_name=best_match.get("customer_name"),
                similarity_score=best_match["similarity"]
            )

            # Log the attempt
            await log_audit(
                db,
                request.session_id,
                "verify",
                "duplicate_found",
                {
                    "matched_customer_id": best_match["customer_id"],
                    "similarity": best_match["similarity"],
                    "review_id": review_id
                },
                http_request
            )

            return VerifyResponse(
                status="duplicate_found",
                message="Your registration is under review. We'll contact you shortly.",
                review_id=review_id,
                can_dispute=False
            )

        # 6. Store the new face
        customer_id = str(uuid.uuid4())

        # Build metadata including motion analysis
        face_metadata = {
            "session_id": request.session_id,
            "challenges": request.challenges_completed,
            "spoof_score": spoof_score,
            "adjusted_spoof_score": adjusted_spoof_score,
        }
        if request.motion_analysis:
            face_metadata["motion_liveness_score"] = request.motion_analysis.liveness_score
            face_metadata["motion_flags"] = request.motion_analysis.flags

        await add_face_embedding(
            customer_id=customer_id,
            embedding=embedding,
            db=db,
            metadata=face_metadata
        )

        # Log success
        await log_audit(
            db,
            request.session_id,
            "verify",
            "success",
            {
                "customer_id": customer_id,
                "spoof_score": spoof_score,
                "adjusted_spoof_score": adjusted_spoof_score,
                "motion_liveness_score": request.motion_analysis.liveness_score if request.motion_analysis else None,
                "challenges": request.challenges_completed
            },
            http_request
        )

        return VerifyResponse(
            status="success",
            message="Verification successful. Your identity has been confirmed.",
            customer_id=customer_id,
            confidence=adjusted_spoof_score
        )

    except Exception as e:
        logger.exception(f"Verification error: {e}")
        return VerifyResponse(
            status="error",
            message="An unexpected error occurred. Please try again.",
            can_dispute=True
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
            :new_embedding::vector,
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
            "new_embedding": str(new_embedding.tolist()),
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
            VALUES (:session_id, :action, :result, :details::jsonb, :ip_address, :user_agent)
        """)

        await db.execute(
            query,
            {
                "session_id": session_id,
                "action": action,
                "result": result,
                "details": str(details).replace("'", '"'),
                "ip_address": ip_address,
                "user_agent": user_agent
            }
        )
        await db.commit()
    except Exception as e:
        logger.error(f"Failed to log audit: {e}")
