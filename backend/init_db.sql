-- KYC Liveness Detection Database Schema
-- Run this in your PostgreSQL database (Neon)

-- Enable pgvector extension
CREATE EXTENSION IF NOT EXISTS vector;

-- Customer faces table (stores verified face embeddings)
CREATE TABLE IF NOT EXISTS customer_faces (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    customer_id VARCHAR(255) NOT NULL UNIQUE,
    customer_name VARCHAR(255),
    embedding vector(512) NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    status VARCHAR(50) DEFAULT 'active',
    metadata JSONB DEFAULT '{}'
);

-- Index for fast similarity search
CREATE INDEX IF NOT EXISTS idx_customer_faces_embedding
ON customer_faces
USING ivfflat (embedding vector_cosine_ops)
WITH (lists = 100);

-- Index for customer_id lookups
CREATE INDEX IF NOT EXISTS idx_customer_faces_customer_id
ON customer_faces (customer_id);

-- Index for status filtering
CREATE INDEX IF NOT EXISTS idx_customer_faces_status
ON customer_faces (status);

-- Review queue (flagged registrations for human review)
CREATE TABLE IF NOT EXISTS review_queue (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),

    -- New registration info
    new_customer_id VARCHAR(255) NOT NULL,
    new_customer_name VARCHAR(255),
    new_face_image_url TEXT,
    new_embedding vector(512),

    -- Matched existing customer
    matched_customer_id VARCHAR(255),
    matched_customer_name VARCHAR(255),
    matched_face_image_url TEXT,
    similarity_score FLOAT NOT NULL,

    -- Review status
    status VARCHAR(50) DEFAULT 'pending',
    reviewed_by VARCHAR(255),
    reviewed_at TIMESTAMP WITH TIME ZONE,
    review_notes TEXT,

    -- Timestamps
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Index for pending reviews
CREATE INDEX IF NOT EXISTS idx_review_queue_status
ON review_queue (status);

-- Index for date sorting
CREATE INDEX IF NOT EXISTS idx_review_queue_created_at
ON review_queue (created_at DESC);

-- Dispute images (stored when users claim they are real)
CREATE TABLE IF NOT EXISTS dispute_images (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    session_id VARCHAR(255) NOT NULL,
    image_url TEXT NOT NULL,
    reason VARCHAR(100) NOT NULL,
    status VARCHAR(50) DEFAULT 'pending',
    reviewed_by VARCHAR(255),
    reviewed_at TIMESTAMP WITH TIME ZONE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Index for pending disputes
CREATE INDEX IF NOT EXISTS idx_dispute_images_status
ON dispute_images (status);

-- Audit log (immutable record of all verification attempts)
CREATE TABLE IF NOT EXISTS audit_log (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    session_id VARCHAR(255) NOT NULL,
    action VARCHAR(100) NOT NULL,
    result VARCHAR(50) NOT NULL,
    details JSONB DEFAULT '{}',
    ip_address INET,
    user_agent TEXT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Index for session lookups
CREATE INDEX IF NOT EXISTS idx_audit_log_session_id
ON audit_log (session_id);

-- Index for date range queries
CREATE INDEX IF NOT EXISTS idx_audit_log_created_at
ON audit_log (created_at DESC);

-- Verify tables were created
SELECT table_name FROM information_schema.tables
WHERE table_schema = 'public'
AND table_name IN ('customer_faces', 'review_queue', 'dispute_images', 'audit_log');
