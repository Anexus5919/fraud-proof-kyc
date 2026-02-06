# Backend Architecture

## Overview

The backend is a FastAPI application that handles:
1. Receiving captured face images
2. Spoof detection (is it a real face or photo/video?)
3. Face embedding extraction
4. Deduplication search against existing faces
5. Storing results and managing review queue

## Technology

- **Python 3.11** - Runtime
- **FastAPI** - Web framework
- **SQLAlchemy** - ORM (async)
- **asyncpg** - PostgreSQL driver
- **Silent-Face-Anti-Spoofing** - Spoof detection model
- **InsightFace** - Face embedding model
- **Cloudinary** - Image storage for disputes

## Project Structure

```
backend/
├── app/
│   ├── __init__.py
│   ├── main.py                 # FastAPI app entry
│   ├── config.py               # Environment variables
│   │
│   ├── routers/
│   │   ├── __init__.py
│   │   ├── verify.py           # /api/verify endpoint
│   │   └── admin.py            # /api/admin/* endpoints
│   │
│   ├── services/
│   │   ├── __init__.py
│   │   ├── spoof_detector.py   # Silent-Face wrapper
│   │   ├── face_embedder.py    # InsightFace wrapper
│   │   ├── deduplication.py    # Vector similarity search
│   │   └── storage.py          # Cloudinary upload
│   │
│   ├── models/
│   │   ├── __init__.py
│   │   ├── database.py         # SQLAlchemy models
│   │   └── schemas.py          # Pydantic schemas
│   │
│   └── db/
│       ├── __init__.py
│       └── session.py          # Database connection
│
├── ml_models/                  # Downloaded model weights
│   ├── silent_face/
│   │   └── anti_spoof_models/
│   └── insightface/
│       └── buffalo_l/
│
├── requirements.txt
├── Dockerfile
└── render.yaml                 # Render deployment config
```

## API Endpoints

### POST /api/verify

Verify a face image for liveness and uniqueness.

**Request:**
```json
{
  "image": "base64_encoded_image_data",
  "session_id": "uuid-string",
  "challenges_completed": ["blink", "smile", "look_left"]
}
```

**Response (Success):**
```json
{
  "status": "success",
  "customer_id": "generated-uuid",
  "message": "Verification successful"
}
```

**Response (Spoof Detected):**
```json
{
  "status": "spoof_detected",
  "confidence": 0.23,
  "message": "We couldn't verify you're physically present. Please try again.",
  "can_dispute": true
}
```

**Response (Duplicate Found):**
```json
{
  "status": "duplicate_found",
  "message": "Your registration is under review. We'll contact you shortly.",
  "review_id": "uuid"
}
```

### POST /api/verify/dispute

Submit dispute when user claims they are real.

**Request:**
```json
{
  "session_id": "uuid",
  "image": "base64_encoded_image",
  "reason": "spoof_detected"
}
```

**Response:**
```json
{
  "status": "submitted",
  "message": "Your dispute has been submitted for manual review."
}
```

### GET /api/admin/reviews

List pending reviews.

**Response:**
```json
{
  "reviews": [
    {
      "id": "uuid",
      "new_customer_name": "John Doe",
      "new_face_url": "https://cloudinary.com/...",
      "matched_customer_name": "Jon Doe",
      "matched_face_url": "https://cloudinary.com/...",
      "similarity_score": 0.87,
      "created_at": "2024-01-15T10:30:00Z"
    }
  ],
  "total": 15
}
```

### POST /api/admin/reviews/:id/approve

Approve registration (different person).

### POST /api/admin/reviews/:id/reject

Reject registration (same person / fraud).

## Processing Pipeline

```
1. RECEIVE IMAGE
   │
   ├─► Decode base64
   ├─► Validate image format
   └─► Check file size (max 5MB)
         │
         ▼
2. FACE DETECTION (InsightFace)
   │
   ├─► No face found? → Return error
   ├─► Multiple faces? → Return error
   └─► Get face bounding box
         │
         ▼
3. SPOOF DETECTION (Silent-Face)
   │
   ├─► Score < 0.5? → Return "spoof_detected"
   └─► Pass (score >= 0.5)
         │
         ▼
4. EMBEDDING EXTRACTION (InsightFace ArcFace)
   │
   └─► Get 512-dimensional vector
         │
         ▼
5. DEDUPLICATION SEARCH (pgvector)
   │
   ├─► Query: SELECT * FROM faces WHERE embedding <=> $1 < 0.4
   ├─► Match found? → Flag for review
   └─► No match? → Store and return success
         │
         ▼
6. STORE RESULT
   │
   ├─► Save embedding to database
   ├─► Log to audit table
   └─► Return response
```

## Spoof Detection Service

Uses Silent-Face-Anti-Spoofing model:

```python
class SpoofDetector:
    def __init__(self):
        self.model = load_model("ml_models/silent_face/anti_spoof_models")

    def detect(self, image: np.ndarray, face_bbox: tuple) -> float:
        """
        Returns spoof score between 0 and 1.
        Higher = more likely real.
        Threshold: 0.5
        """
        # Crop face region
        face = crop_face(image, face_bbox)
        # Run inference
        score = self.model.predict(face)
        return score
```

## Face Embedding Service

Uses InsightFace with ArcFace model:

```python
class FaceEmbedder:
    def __init__(self):
        self.app = insightface.app.FaceAnalysis(
            name='buffalo_l',
            providers=['CPUExecutionProvider']
        )
        self.app.prepare(ctx_id=0)

    def get_embedding(self, image: np.ndarray) -> np.ndarray:
        """
        Returns 512-dimensional face embedding vector.
        """
        faces = self.app.get(image)
        if len(faces) == 0:
            raise NoFaceError()
        return faces[0].embedding
```

## Deduplication Service

Uses pgvector for similarity search:

```python
class DeduplicationService:
    THRESHOLD = 0.4  # Cosine distance threshold

    async def find_duplicates(
        self,
        embedding: np.ndarray,
        db: AsyncSession
    ) -> list[dict]:
        """
        Find faces similar to the given embedding.
        Returns list of matches with similarity scores.
        """
        query = text("""
            SELECT
                id,
                customer_id,
                1 - (embedding <=> :embedding) as similarity
            FROM customer_faces
            WHERE embedding <=> :embedding < :threshold
            ORDER BY embedding <=> :embedding
            LIMIT 5
        """)

        result = await db.execute(
            query,
            {
                "embedding": embedding.tolist(),
                "threshold": self.THRESHOLD
            }
        )
        return result.fetchall()
```

## Environment Variables

```bash
# Database
DATABASE_URL=postgresql+asyncpg://user:pass@host/db

# Cloudinary
CLOUDINARY_CLOUD_NAME=xxx
CLOUDINARY_API_KEY=xxx
CLOUDINARY_API_SECRET=xxx

# CORS
CORS_ORIGINS=https://app.domain.com,https://admin.domain.com

# Optional
LOG_LEVEL=INFO
```

## Model Download

Models are downloaded on first run or during Docker build:

```python
# Silent-Face models
# Clone from: https://github.com/minivision-ai/Silent-Face-Anti-Spoofing
# Copy resources/anti_spoof_models to ml_models/silent_face/

# InsightFace models (auto-download)
# buffalo_l model downloads automatically on first use
# Stored in ~/.insightface/models/buffalo_l/
```

## Error Handling

All errors return consistent format:

```json
{
  "status": "error",
  "error_code": "FACE_NOT_DETECTED",
  "message": "No face was detected in the image. Please ensure your face is clearly visible."
}
```

Error codes:
- `FACE_NOT_DETECTED` - No face in image
- `MULTIPLE_FACES` - More than one face
- `IMAGE_TOO_LARGE` - File size > 5MB
- `INVALID_IMAGE` - Corrupted or unsupported format
- `SPOOF_DETECTED` - Failed liveness check
- `SERVER_ERROR` - Internal error

## Performance Expectations (CPU)

| Operation | Expected Time |
|-----------|---------------|
| Image decode | ~50ms |
| Face detection | ~200ms |
| Spoof detection | ~300ms |
| Embedding extraction | ~400ms |
| Database query | ~50ms |
| **Total** | ~1-1.5 seconds |

Note: First request is slower due to model loading (~5-10 seconds).
