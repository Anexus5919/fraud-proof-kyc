from sqlalchemy import Column, String, Float, DateTime, Text, func
from sqlalchemy.dialects.postgresql import UUID, JSONB
from pgvector.sqlalchemy import Vector
import uuid

from app.db.session import Base


class CustomerFace(Base):
    __tablename__ = "customer_faces"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    customer_id = Column(String(255), nullable=False, unique=True, index=True)
    customer_name = Column(String(255))
    embedding = Column(Vector(512), nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    status = Column(String(50), default="active", index=True)
    metadata = Column(JSONB, default={})


class ReviewQueue(Base):
    __tablename__ = "review_queue"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)

    # New registration info
    new_customer_id = Column(String(255), nullable=False)
    new_customer_name = Column(String(255))
    new_face_image_url = Column(Text)
    new_embedding = Column(Vector(512))

    # Matched existing customer
    matched_customer_id = Column(String(255))
    matched_customer_name = Column(String(255))
    matched_face_image_url = Column(Text)
    similarity_score = Column(Float, nullable=False)

    # Review status
    status = Column(String(50), default="pending", index=True)
    reviewed_by = Column(String(255))
    reviewed_at = Column(DateTime(timezone=True))
    review_notes = Column(Text)

    # Timestamps
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())


class DisputeImage(Base):
    __tablename__ = "dispute_images"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    session_id = Column(String(255), nullable=False)
    image_url = Column(Text, nullable=False)
    reason = Column(String(100), nullable=False)
    status = Column(String(50), default="pending", index=True)
    reviewed_by = Column(String(255))
    reviewed_at = Column(DateTime(timezone=True))
    created_at = Column(DateTime(timezone=True), server_default=func.now())


class AuditLog(Base):
    __tablename__ = "audit_log"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    session_id = Column(String(255), nullable=False, index=True)
    action = Column(String(100), nullable=False)
    result = Column(String(50), nullable=False)
    details = Column(JSONB, default={})
    ip_address = Column(String(45))
    user_agent = Column(Text)
    created_at = Column(DateTime(timezone=True), server_default=func.now(), index=True)
