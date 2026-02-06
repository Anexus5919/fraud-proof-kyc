# Database Schema

## Overview

PostgreSQL database with pgvector extension for face embedding similarity search.

## Provider

**Neon** (neon.tech) - Free tier includes:
- 0.5 GB storage
- pgvector extension
- Serverless (scales to zero)

## Schema

### Enable pgvector

```sql
CREATE EXTENSION IF NOT EXISTS vector;
```

### customer_faces

Stores face embeddings for registered customers.

```sql
CREATE TABLE customer_faces (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    customer_id VARCHAR(255) NOT NULL UNIQUE,
    customer_name VARCHAR(255),
    embedding vector(512) NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    status VARCHAR(50) DEFAULT 'active',
    metadata JSONB DEFAULT '{}'
);

-- Index for fast similarity search (IVFFlat)
-- lists = sqrt(num_rows) is a good starting point
-- For < 100K rows, lists = 100 is fine
CREATE INDEX idx_customer_faces_embedding
ON customer_faces
USING ivfflat (embedding vector_cosine_ops)
WITH (lists = 100);

-- Index for customer_id lookups
CREATE INDEX idx_customer_faces_customer_id
ON customer_faces (customer_id);

-- Index for status filtering
CREATE INDEX idx_customer_faces_status
ON customer_faces (status);
```

### review_queue

Stores flagged registrations for human review.

```sql
CREATE TABLE review_queue (
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
CREATE INDEX idx_review_queue_status
ON review_queue (status);

-- Index for date sorting
CREATE INDEX idx_review_queue_created_at
ON review_queue (created_at DESC);
```

### dispute_images

Stores images when users dispute spoof detection results.

```sql
CREATE TABLE dispute_images (
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
CREATE INDEX idx_dispute_images_status
ON dispute_images (status);
```

### audit_log

Immutable log of all verification attempts.

```sql
CREATE TABLE audit_log (
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
CREATE INDEX idx_audit_log_session_id
ON audit_log (session_id);

-- Index for date range queries
CREATE INDEX idx_audit_log_created_at
ON audit_log (created_at DESC);
```

## Common Queries

### Find similar faces

```sql
-- Find faces within cosine distance threshold
SELECT
    id,
    customer_id,
    customer_name,
    1 - (embedding <=> $1) as similarity
FROM customer_faces
WHERE status = 'active'
  AND embedding <=> $1 < 0.4  -- cosine distance < 0.4 means similarity > 0.6
ORDER BY embedding <=> $1
LIMIT 5;
```

### Insert new face

```sql
INSERT INTO customer_faces (customer_id, customer_name, embedding)
VALUES ($1, $2, $3)
RETURNING id;
```

### Get pending reviews

```sql
SELECT
    r.*,
    cf.customer_name as matched_name
FROM review_queue r
LEFT JOIN customer_faces cf ON cf.customer_id = r.matched_customer_id
WHERE r.status = 'pending'
ORDER BY r.created_at DESC
LIMIT 50;
```

### Approve review (insert face)

```sql
-- Begin transaction
BEGIN;

-- Insert the new face
INSERT INTO customer_faces (customer_id, customer_name, embedding)
VALUES ($1, $2, $3);

-- Update review status
UPDATE review_queue
SET status = 'approved',
    reviewed_by = $4,
    reviewed_at = NOW()
WHERE id = $5;

COMMIT;
```

### Reject review

```sql
UPDATE review_queue
SET status = 'rejected',
    reviewed_by = $1,
    reviewed_at = NOW(),
    review_notes = $2
WHERE id = $3;
```

## Distance Functions

pgvector supports multiple distance functions:

| Operator | Function | Use Case |
|----------|----------|----------|
| `<=>` | Cosine distance | **Recommended for faces** |
| `<->` | L2 (Euclidean) distance | Alternative |
| `<#>` | Inner product (negative) | When vectors are normalized |

For face embeddings, **cosine distance** is standard:
- 0 = identical
- 1 = completely different
- < 0.4 = likely same person

## Index Tuning

### IVFFlat Index

```sql
-- Create index with custom list count
CREATE INDEX ON customer_faces
USING ivfflat (embedding vector_cosine_ops)
WITH (lists = 100);

-- Set probes for query accuracy/speed tradeoff
SET ivfflat.probes = 10;  -- Higher = more accurate, slower
```

### HNSW Index (Alternative)

More accurate but more memory:

```sql
CREATE INDEX ON customer_faces
USING hnsw (embedding vector_cosine_ops)
WITH (m = 16, ef_construction = 64);
```

## Migrations

For hackathon, run schema manually in Neon SQL Editor.

For production, use Alembic:

```bash
# Generate migration
alembic revision --autogenerate -m "description"

# Run migrations
alembic upgrade head
```

## Backup

Neon provides automatic point-in-time recovery on free tier.

For manual backup:
```bash
pg_dump $DATABASE_URL > backup.sql
```
