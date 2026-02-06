import logging
from typing import Optional
from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import text
from datetime import datetime

from app.db.session import get_db
from app.models.schemas import (
    ReviewListResponse,
    ReviewItem,
    ReviewActionRequest,
    ReviewActionResponse,
    DisputeListResponse,
    DisputeItem
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/admin", tags=["admin"])


@router.get("/reviews", response_model=ReviewListResponse)
async def list_reviews(
    status: Optional[str] = Query("pending", description="Filter by status"),
    limit: int = Query(50, le=100),
    offset: int = Query(0),
    db: AsyncSession = Depends(get_db)
):
    """List pending reviews for duplicates"""
    try:
        # Build query based on whether status is provided
        if status:
            count_query = text("SELECT COUNT(*) FROM review_queue WHERE status = :status")
            count_result = await db.execute(count_query, {"status": status})
        else:
            count_query = text("SELECT COUNT(*) FROM review_queue")
            count_result = await db.execute(count_query)
        total = count_result.scalar()

        # Get reviews
        if status:
            query = text("""
                SELECT
                    id,
                    new_customer_id,
                    new_customer_name,
                    new_face_image_url,
                    matched_customer_id,
                    matched_customer_name,
                    matched_face_image_url,
                    similarity_score,
                    status,
                    created_at
                FROM review_queue
                WHERE status = :status
                ORDER BY created_at DESC
                LIMIT :limit OFFSET :offset
            """)
        else:
            query = text("""
                SELECT
                    id,
                    new_customer_id,
                    new_customer_name,
                    new_face_image_url,
                    matched_customer_id,
                    matched_customer_name,
                    matched_face_image_url,
                    similarity_score,
                    status,
                    created_at
                FROM review_queue
                ORDER BY created_at DESC
                LIMIT :limit OFFSET :offset
            """)

        if status:
            result = await db.execute(query, {"status": status, "limit": limit, "offset": offset})
        else:
            result = await db.execute(query, {"limit": limit, "offset": offset})

        reviews = []
        for row in result:
            reviews.append(ReviewItem(
                id=row.id,
                new_customer_id=row.new_customer_id,
                new_customer_name=row.new_customer_name,
                new_face_image_url=row.new_face_image_url,
                matched_customer_id=row.matched_customer_id,
                matched_customer_name=row.matched_customer_name,
                matched_face_image_url=row.matched_face_image_url,
                similarity_score=row.similarity_score,
                status=row.status,
                created_at=row.created_at
            ))

        return ReviewListResponse(reviews=reviews, total=total)

    except Exception as e:
        logger.exception(f"Error listing reviews: {e}")
        raise HTTPException(status_code=500, detail="Failed to list reviews")


@router.get("/reviews/{review_id}")
async def get_review(
    review_id: str,
    db: AsyncSession = Depends(get_db)
):
    """Get details of a specific review"""
    query = text("""
        SELECT
            r.*,
            cf.customer_name as matched_name_from_db
        FROM review_queue r
        LEFT JOIN customer_faces cf ON cf.customer_id = r.matched_customer_id
        WHERE r.id = CAST(:review_id AS uuid)
    """)

    result = await db.execute(query, {"review_id": review_id})
    row = result.fetchone()

    if not row:
        raise HTTPException(status_code=404, detail="Review not found")

    return {
        "id": str(row.id),
        "new_customer_id": row.new_customer_id,
        "new_customer_name": row.new_customer_name,
        "new_face_image_url": row.new_face_image_url,
        "matched_customer_id": row.matched_customer_id,
        "matched_customer_name": row.matched_customer_name or row.matched_name_from_db,
        "matched_face_image_url": row.matched_face_image_url,
        "similarity_score": row.similarity_score,
        "status": row.status,
        "reviewed_by": row.reviewed_by,
        "reviewed_at": row.reviewed_at,
        "review_notes": row.review_notes,
        "created_at": row.created_at
    }


@router.post("/reviews/{review_id}/approve", response_model=ReviewActionResponse)
async def approve_review(
    review_id: str,
    request: ReviewActionRequest,
    db: AsyncSession = Depends(get_db)
):
    """
    Approve a review - marks the new registration as a different person.
    The new face will be added to the database.
    """
    try:
        # Get review details
        get_query = text("""
            SELECT * FROM review_queue WHERE id = CAST(:review_id AS uuid) AND status = 'pending'
        """)
        result = await db.execute(get_query, {"review_id": review_id})
        review = result.fetchone()

        if not review:
            raise HTTPException(status_code=404, detail="Review not found or already processed")

        # Add the new face to database
        # The embedding comes from review_queue as a pgvector type — pass it through
        # using CAST to ensure correct type handling with asyncpg
        insert_query = text("""
            INSERT INTO customer_faces (customer_id, customer_name, embedding, metadata)
            SELECT
                :customer_id,
                :customer_name,
                rq.new_embedding,
                CAST('{"from_review": true}' AS jsonb)
            FROM review_queue rq
            WHERE rq.id = CAST(:review_id AS uuid)
        """)

        await db.execute(insert_query, {
            "customer_id": review.new_customer_id,
            "customer_name": review.new_customer_name,
            "review_id": review_id
        })

        # Update review status
        update_query = text("""
            UPDATE review_queue
            SET status = 'approved',
                reviewed_by = 'admin',
                reviewed_at = :now,
                review_notes = :notes
            WHERE id = CAST(:review_id AS uuid)
        """)

        await db.execute(update_query, {
            "review_id": review_id,
            "now": datetime.utcnow(),
            "notes": request.notes
        })

        await db.commit()

        return ReviewActionResponse(
            status="success",
            message="Review approved. The customer has been registered."
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"Error approving review: {e}")
        raise HTTPException(status_code=500, detail="Failed to approve review")


@router.post("/reviews/{review_id}/reject", response_model=ReviewActionResponse)
async def reject_review(
    review_id: str,
    request: ReviewActionRequest,
    db: AsyncSession = Depends(get_db)
):
    """
    Reject a review - confirms this is a duplicate/fraud attempt.
    """
    try:
        # Update review status
        update_query = text("""
            UPDATE review_queue
            SET status = 'rejected',
                reviewed_by = 'admin',
                reviewed_at = :now,
                review_notes = :notes
            WHERE id = CAST(:review_id AS uuid) AND status = 'pending'
            RETURNING id
        """)

        result = await db.execute(update_query, {
            "review_id": review_id,
            "now": datetime.utcnow(),
            "notes": request.notes
        })

        row = result.fetchone()
        if not row:
            raise HTTPException(status_code=404, detail="Review not found or already processed")

        await db.commit()

        return ReviewActionResponse(
            status="success",
            message="Review rejected. This has been marked as a duplicate/fraud attempt."
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"Error rejecting review: {e}")
        raise HTTPException(status_code=500, detail="Failed to reject review")


@router.get("/disputes", response_model=DisputeListResponse)
async def list_disputes(
    status: Optional[str] = Query("pending", description="Filter by status"),
    limit: int = Query(50, le=100),
    offset: int = Query(0),
    db: AsyncSession = Depends(get_db)
):
    """List disputes submitted by users"""
    try:
        # Count total
        if status:
            count_query = text("SELECT COUNT(*) FROM dispute_images WHERE status = :status")
            count_result = await db.execute(count_query, {"status": status})
        else:
            count_query = text("SELECT COUNT(*) FROM dispute_images")
            count_result = await db.execute(count_query)
        total = count_result.scalar()

        # Get disputes
        if status:
            query = text("""
                SELECT id, session_id, image_url, reason, status, created_at
                FROM dispute_images
                WHERE status = :status
                ORDER BY created_at DESC
                LIMIT :limit OFFSET :offset
            """)
            result = await db.execute(query, {"status": status, "limit": limit, "offset": offset})
        else:
            query = text("""
                SELECT id, session_id, image_url, reason, status, created_at
                FROM dispute_images
                ORDER BY created_at DESC
                LIMIT :limit OFFSET :offset
            """)
            result = await db.execute(query, {"limit": limit, "offset": offset})

        disputes = []
        for row in result:
            disputes.append(DisputeItem(
                id=row.id,
                session_id=row.session_id,
                image_url=row.image_url,
                reason=row.reason,
                status=row.status,
                created_at=row.created_at
            ))

        return DisputeListResponse(disputes=disputes, total=total)

    except Exception as e:
        logger.exception(f"Error listing disputes: {e}")
        raise HTTPException(status_code=500, detail="Failed to list disputes")


@router.get("/stats")
async def get_stats(db: AsyncSession = Depends(get_db)):
    """Get dashboard statistics"""
    try:
        stats = {}

        # Total customers
        result = await db.execute(text("SELECT COUNT(*) FROM customer_faces WHERE status = 'active'"))
        stats["total_customers"] = result.scalar()

        # Pending reviews
        result = await db.execute(text("SELECT COUNT(*) FROM review_queue WHERE status = 'pending'"))
        stats["pending_reviews"] = result.scalar()

        # Pending disputes
        result = await db.execute(text("SELECT COUNT(*) FROM dispute_images WHERE status = 'pending'"))
        stats["pending_disputes"] = result.scalar()

        # Today's verifications
        result = await db.execute(text("""
            SELECT COUNT(*) FROM audit_log
            WHERE action = 'verify' AND created_at >= CURRENT_DATE
        """))
        stats["today_verifications"] = result.scalar()

        return stats

    except Exception as e:
        logger.exception(f"Error getting stats: {e}")
        raise HTTPException(status_code=500, detail="Failed to get stats")
