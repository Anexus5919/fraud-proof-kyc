import json
import numpy as np
from typing import List, Optional
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import text
import logging

from app.config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()


async def find_similar_faces(
    embedding: np.ndarray,
    db: AsyncSession,
    threshold: Optional[float] = None,
    limit: int = 5
) -> List[dict]:
    """
    Find faces similar to the given embedding using cosine distance.

    Args:
        embedding: 512-dim face embedding vector
        db: Database session
        threshold: Cosine distance threshold (lower = more similar)
        limit: Maximum number of results

    Returns:
        List of matches with customer_id, similarity score, etc.
    """
    if threshold is None:
        threshold = settings.duplicate_threshold

    logger.info(f"[DEDUP] Searching for similar faces — threshold={threshold}, limit={limit}, "
               f"embedding_norm={float(np.linalg.norm(embedding)):.4f}")

    # Convert embedding to PostgreSQL vector string format
    embedding_str = '[' + ','.join(map(str, embedding.tolist())) + ']'

    # First check how many active faces exist in DB
    count_result = await db.execute(text("SELECT COUNT(*) FROM customer_faces WHERE status = 'active'"))
    active_count = count_result.scalar()
    logger.info(f"[DEDUP] Active faces in DB: {active_count}")

    # Query using pgvector cosine distance operator
    # Use CAST instead of :: to avoid asyncpg parameter binding issues
    query = text("""
        SELECT
            id,
            customer_id,
            customer_name,
            1 - (embedding <=> CAST(:embedding AS vector)) as similarity,
            embedding <=> CAST(:embedding AS vector) as distance
        FROM customer_faces
        WHERE status = 'active'
          AND embedding <=> CAST(:embedding AS vector) < :threshold
        ORDER BY embedding <=> CAST(:embedding AS vector)
        LIMIT :limit
    """)

    result = await db.execute(
        query,
        {
            "embedding": embedding_str,
            "threshold": threshold,
            "limit": limit
        }
    )

    matches = []
    for row in result:
        matches.append({
            "id": str(row.id),
            "customer_id": row.customer_id,
            "customer_name": row.customer_name,
            "similarity": float(row.similarity),
            "distance": float(row.distance)
        })

    if matches:
        logger.info(f"[DEDUP] Found {len(matches)} match(es):")
        for i, m in enumerate(matches):
            logger.info(f"  [{i+1}] customer={m['customer_id']}, name={m['customer_name']}, "
                       f"similarity={m['similarity']:.4f}, distance={m['distance']:.4f}")
    else:
        logger.info("[DEDUP] No duplicates found — face is unique")

    return matches


async def add_face_embedding(
    customer_id: str,
    embedding: np.ndarray,
    db: AsyncSession,
    customer_name: Optional[str] = None,
    metadata: Optional[dict] = None,
    face_image: Optional[str] = None
) -> str:
    """
    Add a new face embedding to the database.

    Args:
        customer_id: Unique customer identifier
        embedding: 512-dim face embedding vector
        db: Database session
        customer_name: Optional customer name
        metadata: Optional metadata dict
        face_image: Optional base64 data URI of the face image

    Returns:
        ID of the inserted record
    """
    logger.info(f"[DEDUP] Storing embedding — customer_id={customer_id}, name={customer_name}, has_image={face_image is not None}")

    embedding_str = '[' + ','.join(map(str, embedding.tolist())) + ']'
    metadata_str = json.dumps(metadata or {}, default=str)

    query = text("""
        INSERT INTO customer_faces (customer_id, customer_name, embedding, metadata, face_image)
        VALUES (:customer_id, :customer_name, CAST(:embedding AS vector), CAST(:metadata AS jsonb), :face_image)
        RETURNING id
    """)

    result = await db.execute(
        query,
        {
            "customer_id": customer_id,
            "customer_name": customer_name,
            "embedding": embedding_str,
            "metadata": metadata_str,
            "face_image": face_image
        }
    )

    row = result.fetchone()
    await db.commit()

    logger.info(f"[DEDUP] Stored successfully — record_id={row.id}")
    return str(row.id)


async def check_duplicate(
    embedding: np.ndarray,
    db: AsyncSession,
    threshold: Optional[float] = None
) -> tuple[bool, Optional[dict]]:
    """
    Check if a face embedding has duplicates in the database.

    Args:
        embedding: 512-dim face embedding vector
        db: Database session
        threshold: Cosine distance threshold

    Returns:
        Tuple of (is_duplicate, best_match or None)
    """
    matches = await find_similar_faces(embedding, db, threshold, limit=1)

    if matches:
        return True, matches[0]

    return False, None
