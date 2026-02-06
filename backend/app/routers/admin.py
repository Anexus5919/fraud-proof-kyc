import json
import logging
from typing import Optional
from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import text
from datetime import datetime

from app.db.session import get_db

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/admin", tags=["admin"])


@router.get("/reviews")
async def list_reviews(
    status: Optional[str] = Query("pending", description="Filter by status"),
    limit: int = Query(50, le=100),
    offset: int = Query(0),
    db: AsyncSession = Depends(get_db)
):
    """List reviews with telemetry data from audit_log"""
    try:
        params = {"limit": limit, "offset": offset}
        where_clause = ""
        if status:
            where_clause = "WHERE r.status = :status"
            params["status"] = status

        count_query = text(f"SELECT COUNT(*) FROM review_queue r {where_clause}")
        count_result = await db.execute(count_query, params)
        total = count_result.scalar()

        query = text(f"""
            SELECT
                r.id,
                r.new_customer_id,
                r.new_customer_name,
                r.new_face_image_url,
                r.matched_customer_id,
                r.matched_customer_name,
                r.matched_face_image_url,
                r.similarity_score,
                r.status,
                r.created_at,
                r.reviewed_by,
                r.reviewed_at,
                r.review_notes,
                a.details as audit_details
            FROM review_queue r
            LEFT JOIN LATERAL (
                SELECT details FROM audit_log
                WHERE session_id = r.new_customer_id AND action = 'verify'
                ORDER BY created_at DESC LIMIT 1
            ) a ON true
            {where_clause}
            ORDER BY r.created_at DESC
            LIMIT :limit OFFSET :offset
        """)

        result = await db.execute(query, params)

        reviews = []
        for row in result:
            audit = row.audit_details or {}
            reviews.append({
                "id": str(row.id),
                "new_customer_id": row.new_customer_id,
                "new_customer_name": row.new_customer_name,
                "new_face_image_url": row.new_face_image_url,
                "matched_customer_id": row.matched_customer_id,
                "matched_customer_name": row.matched_customer_name,
                "matched_face_image_url": row.matched_face_image_url,
                "similarity_score": row.similarity_score,
                "status": row.status,
                "created_at": row.created_at.isoformat() if row.created_at else None,
                "reviewed_by": row.reviewed_by,
                "reviewed_at": row.reviewed_at.isoformat() if row.reviewed_at else None,
                "review_notes": row.review_notes,
                "risk_score": audit.get("risk_score"),
                "spoof_score": audit.get("spoof_score"),
                "deepfake_score": audit.get("deepfake_score"),
                "flags": audit.get("flags", []),
            })

        return {"reviews": reviews, "total": total}

    except Exception as e:
        logger.exception(f"Error listing reviews: {e}")
        raise HTTPException(status_code=500, detail="Failed to list reviews")


@router.get("/reviews/{review_id}")
async def get_review(
    review_id: str,
    db: AsyncSession = Depends(get_db)
):
    """Get details of a specific review with telemetry"""
    query = text("""
        SELECT
            r.*,
            cf.customer_name as matched_name_from_db,
            cf.face_image as matched_face_from_db,
            a.details as audit_details
        FROM review_queue r
        LEFT JOIN customer_faces cf ON cf.customer_id = r.matched_customer_id
        LEFT JOIN LATERAL (
            SELECT details FROM audit_log
            WHERE session_id = r.new_customer_id AND action = 'verify'
            ORDER BY created_at DESC LIMIT 1
        ) a ON true
        WHERE r.id = CAST(:review_id AS uuid)
    """)

    result = await db.execute(query, {"review_id": review_id})
    row = result.fetchone()

    if not row:
        raise HTTPException(status_code=404, detail="Review not found")

    audit = row.audit_details or {}

    # Use matched face from DB if review_queue doesn't have it
    matched_image = row.matched_face_image_url or row.matched_face_from_db

    return {
        "id": str(row.id),
        "new_customer_id": row.new_customer_id,
        "new_customer_name": row.new_customer_name,
        "new_face_image_url": row.new_face_image_url,
        "matched_customer_id": row.matched_customer_id,
        "matched_customer_name": row.matched_customer_name or row.matched_name_from_db,
        "matched_face_image_url": matched_image,
        "similarity_score": row.similarity_score,
        "status": row.status,
        "reviewed_by": row.reviewed_by,
        "reviewed_at": row.reviewed_at,
        "review_notes": row.review_notes,
        "created_at": row.created_at,
        "risk_score": audit.get("risk_score"),
        "spoof_score": audit.get("spoof_score"),
        "deepfake_score": audit.get("deepfake_score"),
        "flags": audit.get("flags", []),
    }


@router.post("/reviews/{review_id}/approve")
async def approve_review(
    review_id: str,
    db: AsyncSession = Depends(get_db)
):
    """
    Approve a review - marks the new registration as a different person.
    The new face will be added to the database with its image.
    """
    try:
        get_query = text("""
            SELECT * FROM review_queue WHERE id = CAST(:review_id AS uuid) AND status = 'pending'
        """)
        result = await db.execute(get_query, {"review_id": review_id})
        review = result.fetchone()

        if not review:
            raise HTTPException(status_code=404, detail="Review not found or already processed")

        # Insert into customer_faces with face_image from review_queue
        insert_query = text("""
            INSERT INTO customer_faces (customer_id, customer_name, embedding, face_image, metadata)
            SELECT
                :customer_id,
                :customer_name,
                rq.new_embedding,
                rq.new_face_image_url,
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
            "notes": "Approved by admin"
        })

        await db.commit()

        return {
            "status": "success",
            "message": "Review approved. The customer has been registered with their face image."
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"Error approving review: {e}")
        raise HTTPException(status_code=500, detail="Failed to approve review")


@router.post("/reviews/{review_id}/reject")
async def reject_review(
    review_id: str,
    db: AsyncSession = Depends(get_db)
):
    """
    Reject a review - confirms this is a duplicate/fraud attempt.
    """
    try:
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
            "notes": "Rejected as duplicate by admin"
        })

        row = result.fetchone()
        if not row:
            raise HTTPException(status_code=404, detail="Review not found or already processed")

        await db.commit()

        return {
            "status": "success",
            "message": "Review rejected. This has been marked as a duplicate/fraud attempt."
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"Error rejecting review: {e}")
        raise HTTPException(status_code=500, detail="Failed to reject review")


@router.get("/customers")
async def list_customers(
    status: Optional[str] = Query("active", description="Filter by status"),
    limit: int = Query(50, le=100),
    offset: int = Query(0),
    db: AsyncSession = Depends(get_db)
):
    """List all registered customers with face images and metadata"""
    try:
        params = {"limit": limit, "offset": offset}
        where_clause = ""
        if status:
            where_clause = "WHERE status = :status"
            params["status"] = status

        count_query = text(f"SELECT COUNT(*) FROM customer_faces {where_clause}")
        count_result = await db.execute(count_query, params)
        total = count_result.scalar()

        query = text(f"""
            SELECT
                id, customer_id, customer_name, face_image,
                created_at, status, metadata
            FROM customer_faces
            {where_clause}
            ORDER BY created_at DESC
            LIMIT :limit OFFSET :offset
        """)

        result = await db.execute(query, params)

        customers = []
        for row in result:
            meta = row.metadata or {}
            customers.append({
                "id": str(row.id),
                "customer_id": row.customer_id,
                "customer_name": row.customer_name,
                "face_image": row.face_image,
                "created_at": row.created_at.isoformat() if row.created_at else None,
                "status": row.status,
                "risk_score": meta.get("risk_score"),
                "spoof_score": meta.get("spoof_score"),
                "deepfake_score": meta.get("deepfake_score"),
                "session_id": meta.get("session_id"),
            })

        return {"customers": customers, "total": total}

    except Exception as e:
        logger.exception(f"Error listing customers: {e}")
        raise HTTPException(status_code=500, detail="Failed to list customers")


@router.get("/customers/{customer_id}")
async def get_customer(
    customer_id: str,
    db: AsyncSession = Depends(get_db)
):
    """Get customer detail with face image and audit trail"""
    try:
        # Get customer
        customer_query = text("""
            SELECT id, customer_id, customer_name, face_image, created_at, status, metadata
            FROM customer_faces
            WHERE customer_id = :customer_id
        """)
        result = await db.execute(customer_query, {"customer_id": customer_id})
        customer = result.fetchone()

        if not customer:
            raise HTTPException(status_code=404, detail="Customer not found")

        meta = customer.metadata or {}

        # Get audit trail for this customer
        audit_query = text("""
            SELECT id, session_id, action, result, details, ip_address, user_agent, created_at
            FROM audit_log
            WHERE details ->> 'customer_id' = :customer_id
            ORDER BY created_at DESC
            LIMIT 20
        """)
        audit_result = await db.execute(audit_query, {"customer_id": customer_id})

        audit_trail = []
        for row in audit_result:
            audit_trail.append({
                "id": str(row.id),
                "session_id": row.session_id,
                "action": row.action,
                "result": row.result,
                "details": row.details,
                "ip_address": str(row.ip_address) if row.ip_address else None,
                "created_at": row.created_at.isoformat() if row.created_at else None,
            })

        return {
            "id": str(customer.id),
            "customer_id": customer.customer_id,
            "customer_name": customer.customer_name,
            "face_image": customer.face_image,
            "created_at": customer.created_at.isoformat() if customer.created_at else None,
            "status": customer.status,
            "risk_score": meta.get("risk_score"),
            "spoof_score": meta.get("spoof_score"),
            "deepfake_score": meta.get("deepfake_score"),
            "session_id": meta.get("session_id"),
            "challenges": meta.get("challenges"),
            "metadata": meta,
            "audit_trail": audit_trail,
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"Error getting customer: {e}")
        raise HTTPException(status_code=500, detail="Failed to get customer")


@router.get("/verifications")
async def list_verifications(
    result_filter: Optional[str] = Query(None, alias="result", description="Filter by result"),
    limit: int = Query(50, le=100),
    offset: int = Query(0),
    db: AsyncSession = Depends(get_db)
):
    """List all verification attempts from audit log"""
    try:
        params = {"limit": limit, "offset": offset}
        where_clause = "WHERE action = 'verify'"
        if result_filter:
            where_clause += " AND result = :result_filter"
            params["result_filter"] = result_filter

        count_query = text(f"SELECT COUNT(*) FROM audit_log {where_clause}")
        count_result = await db.execute(count_query, params)
        total = count_result.scalar()

        query = text(f"""
            SELECT id, session_id, action, result, details, ip_address, user_agent, created_at
            FROM audit_log
            {where_clause}
            ORDER BY created_at DESC
            LIMIT :limit OFFSET :offset
        """)

        result = await db.execute(query, params)

        verifications = []
        for row in result:
            details = row.details or {}
            verifications.append({
                "id": str(row.id),
                "session_id": row.session_id,
                "result": row.result,
                "risk_score": details.get("risk_score"),
                "spoof_score": details.get("spoof_score"),
                "deepfake_score": details.get("deepfake_score"),
                "customer_id": details.get("customer_id"),
                "flags": details.get("flags", []),
                "ip_address": str(row.ip_address) if row.ip_address else None,
                "created_at": row.created_at.isoformat() if row.created_at else None,
            })

        return {"verifications": verifications, "total": total}

    except Exception as e:
        logger.exception(f"Error listing verifications: {e}")
        raise HTTPException(status_code=500, detail="Failed to list verifications")


@router.get("/disputes")
async def list_disputes(
    status: Optional[str] = Query("pending", description="Filter by status"),
    limit: int = Query(50, le=100),
    offset: int = Query(0),
    db: AsyncSession = Depends(get_db)
):
    """List disputes submitted by users"""
    try:
        params = {"limit": limit, "offset": offset}
        where_clause = ""
        if status:
            where_clause = "WHERE status = :status"
            params["status"] = status

        count_query = text(f"SELECT COUNT(*) FROM dispute_images {where_clause}")
        count_result = await db.execute(count_query, params)
        total = count_result.scalar()

        query = text(f"""
            SELECT id, session_id, image_url, reason, status, created_at
            FROM dispute_images
            {where_clause}
            ORDER BY created_at DESC
            LIMIT :limit OFFSET :offset
        """)

        result = await db.execute(query, params)

        disputes = []
        for row in result:
            disputes.append({
                "id": str(row.id),
                "session_id": row.session_id,
                "image_url": row.image_url,
                "reason": row.reason,
                "status": row.status,
                "created_at": row.created_at.isoformat() if row.created_at else None,
            })

        return {"disputes": disputes, "total": total}

    except Exception as e:
        logger.exception(f"Error listing disputes: {e}")
        raise HTTPException(status_code=500, detail="Failed to list disputes")


@router.get("/stats")
async def get_stats(db: AsyncSession = Depends(get_db)):
    """Get dashboard statistics"""
    try:
        stats = {}

        result = await db.execute(text("SELECT COUNT(*) FROM customer_faces WHERE status = 'active'"))
        stats["total_customers"] = result.scalar()

        result = await db.execute(text("SELECT COUNT(*) FROM review_queue WHERE status = 'pending'"))
        stats["pending_reviews"] = result.scalar()

        result = await db.execute(text("SELECT COUNT(*) FROM dispute_images WHERE status = 'pending'"))
        stats["pending_disputes"] = result.scalar()

        result = await db.execute(text("""
            SELECT COUNT(*) FROM audit_log
            WHERE action = 'verify' AND created_at >= CURRENT_DATE
        """))
        stats["today_verifications"] = result.scalar()

        # Total verifications ever
        result = await db.execute(text("SELECT COUNT(*) FROM audit_log WHERE action = 'verify'"))
        stats["total_verifications"] = result.scalar()

        return stats

    except Exception as e:
        logger.exception(f"Error getting stats: {e}")
        raise HTTPException(status_code=500, detail="Failed to get stats")
