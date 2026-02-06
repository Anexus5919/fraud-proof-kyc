# KYC Liveness Detection System

## Project Overview

A fraud-proof KYC system that ensures unique customer identities through:
1. **Liveness Detection** - Verify the person is physically present (not a photo/video)
2. **Facial Deduplication** - Ensure the same face isn't registered multiple times

## Core Constraints

### Technical Constraints
- **Browser-based capture** - WebRTC for camera access, no native app
- **Free tier only** - Hackathon prototype, zero cost deployment
- **CPU inference** - No GPU required, acceptable latency for demo
- **Single database** - PostgreSQL with pgvector for everything

### Business Constraints
- **India compliance context** - RBI KYC guidelines awareness
- **Strict liveness** - 3 random challenges per session
- **No blocking on retry** - Users can retry unlimited times
- **Human review for duplicates** - Flag, don't auto-reject

### Design Constraints (from DESIGN_GUIDE.md)
- Intentional over decorative
- Calm confidence, not visual noise
- Human-first, not machine-generated
- Clear visual hierarchy
- Restrained color palette
- Purposeful micro-interactions only

## Architecture

```
┌─────────────────────────────────────────┐
│         app.domain.com (Vercel)         │
│              React + Vite               │
│         MediaPipe Face Landmarker       │
└─────────────────┬───────────────────────┘
                  │
                  ▼
┌─────────────────────────────────────────┐
│        api.domain.com (Render)          │
│              FastAPI                    │
│     Silent-Face + InsightFace           │
└─────────────────┬───────────────────────┘
                  │
        ┌─────────┴─────────┐
        ▼                   ▼
┌──────────────┐    ┌──────────────┐
│     Neon     │    │  Cloudinary  │
│  PostgreSQL  │    │   (images)   │
│  + pgvector  │    │              │
└──────────────┘    └──────────────┘

┌─────────────────────────────────────────┐
│       admin.domain.com (Vercel)         │
│         React Admin Dashboard           │
└─────────────────────────────────────────┘
```

## Tech Stack

### Frontend (Main App)
- React 18 with Vite
- MediaPipe Face Landmarker (via CDN)
- Tailwind CSS (utility-first, matches design guide)
- No component library (custom components for intentional design)

### Frontend (Admin)
- React 18 with Vite
- Tailwind CSS
- Simple table-based review interface

### Backend
- Python 3.11
- FastAPI
- Silent-Face-Anti-Spoofing (spoof detection)
- InsightFace (face embedding)
- SQLAlchemy + asyncpg (database)
- Cloudinary SDK (image storage)

### Database
- PostgreSQL 16 (Neon free tier)
- pgvector extension for similarity search

## Project Structure

```
kyc-liveness/
├── CLAUDE.md                 # This file
├── DESIGN_GUIDE.md           # Design principles
├── docs/
│   ├── FRONTEND.md           # Frontend architecture
│   ├── BACKEND.md            # Backend architecture
│   ├── DATABASE.md           # Schema and queries
│   ├── CHALLENGES.md         # Liveness challenge specs
│   └── DEPLOYMENT.md         # Deployment guide
├── frontend/                 # Main React app
├── admin/                    # Admin React app
└── backend/                  # FastAPI server
```

## Key Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| State management | React useState + useReducer | Simple app, no need for Redux |
| Styling | Tailwind CSS | Matches design guide, utility-first |
| Face detection | MediaPipe | Free, runs in browser, has blendshapes |
| Spoof detection | Silent-Face | Open source, proven, lightweight |
| Embeddings | InsightFace ArcFace | State-of-art accuracy |
| Vector search | pgvector | Simplicity, free with Neon |

## User Flow States

```
IDLE
  ↓
INITIALIZING (loading MediaPipe)
  ↓
QUALITY_CHECK (face position, lighting, obstructions)
  ↓
CHALLENGE_1 (random from pool)
  ↓
CHALLENGE_2 (random, different from 1)
  ↓
CHALLENGE_3 (random, different from 1,2)
  ↓
CAPTURING (best frame selection)
  ↓
PROCESSING (server verification)
  ↓
SUCCESS / FAILURE / FLAGGED_FOR_REVIEW
```

## Quality Checks (Before Challenges)

1. Face detected
2. Single face only
3. Face centered (within bounds)
4. Good lighting (not too dark/bright)
5. Eyes visible (no sunglasses)
6. Nose visible (no mask)
7. Mouth visible (no mask)

## Challenge Pool

| ID | Name | Blendshapes | Threshold | Hold Frames |
|----|------|-------------|-----------|-------------|
| 1 | Blink | eyeBlinkLeft, eyeBlinkRight | > 0.6 | 3 |
| 2 | Smile | mouthSmileLeft, mouthSmileRight | > 0.5 | 5 |
| 3 | Open Mouth | jawOpen | > 0.6 | 3 |
| 4 | Raise Eyebrows | browInnerUp | > 0.5 | 3 |
| 5 | Look Left | calculated from landmarks | - | 3 |
| 6 | Look Right | calculated from landmarks | - | 3 |

## API Endpoints

### Main App
- `POST /api/verify` - Submit face for verification

### Admin
- `GET /api/admin/reviews` - List pending reviews
- `GET /api/admin/reviews/:id` - Get review details
- `POST /api/admin/reviews/:id/approve` - Approve registration
- `POST /api/admin/reviews/:id/reject` - Reject as duplicate

## Environment Variables

### Frontend
- `VITE_API_URL` - Backend API URL

### Backend
- `DATABASE_URL` - Neon PostgreSQL connection string
- `CLOUDINARY_URL` - Cloudinary credentials
- `CORS_ORIGINS` - Allowed frontend origins

## Commands

### Frontend
```bash
cd frontend
npm install
npm run dev      # Development
npm run build    # Production build
```

### Admin
```bash
cd admin
npm install
npm run dev      # Development
npm run build    # Production build
```

### Backend
```bash
cd backend
pip install -r requirements.txt
uvicorn app.main:app --reload  # Development
```

## Testing Checklist

- [ ] Camera permissions work
- [ ] Face detection works in various lighting
- [ ] Glasses detection triggers warning
- [ ] All 6 challenges work correctly
- [ ] Challenges are randomized
- [ ] Backend spoof detection works
- [ ] Duplicate detection works
- [ ] Admin review flow works
- [ ] Dispute image upload works
