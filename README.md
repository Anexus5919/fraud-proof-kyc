# KYC Liveness Detection System

A fraud-proof KYC system that ensures unique customer identities through liveness detection and facial deduplication.

## Features

- **Liveness Detection**: Active challenges (blink, smile, head turn) to verify physical presence
- **Facial Deduplication**: Detect if the same face is registered multiple times
- **Anti-Spoofing**: Detect photos, videos, and other spoofing attempts
- **Admin Dashboard**: Review flagged registrations and disputes

## Architecture

```
Frontend (React) → Backend (FastAPI) → PostgreSQL (pgvector)
     ↑                    ↓
MediaPipe         InsightFace + Silent-Face
```

## Quick Start

### Prerequisites

- Node.js 18+
- Python 3.11+
- Docker (for local PostgreSQL)

### 1. Start Database

```bash
docker-compose up -d
```

This starts PostgreSQL with pgvector and initializes the schema.

### 2. Start Backend

```bash
cd backend
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt

# Set environment variables
export DATABASE_URL="postgresql+asyncpg://kyc:localdev@localhost:5432/kyc_db"
export CORS_ORIGINS="http://localhost:5173,http://localhost:5174"

# Run
uvicorn app.main:app --reload --port 8000
```

### 3. Start Frontend

```bash
cd frontend
npm install
echo "VITE_API_URL=http://localhost:8000" > .env.local
npm run dev
```

Frontend runs at http://localhost:5173

### 4. Start Admin Dashboard

```bash
cd admin
npm install
echo "VITE_API_URL=http://localhost:8000" > .env.local
npm run dev
```

Admin runs at http://localhost:5174

## Project Structure

```
kyc-liveness/
├── frontend/          # User-facing React app
├── admin/             # Admin dashboard React app
├── backend/           # FastAPI backend
├── docs/              # Documentation
├── CLAUDE.md          # Project constraints & architecture
└── docker-compose.yml # Local development setup
```

## Liveness Challenges

The system uses 3 random challenges from:

1. **Blink** - User must blink both eyes
2. **Smile** - User must smile
3. **Open Mouth** - User must open mouth wide
4. **Raise Eyebrows** - User must raise eyebrows
5. **Turn Left** - User must turn head left
6. **Turn Right** - User must turn head right

## Deployment

See [docs/DEPLOYMENT.md](docs/DEPLOYMENT.md) for deployment instructions using:

- **Frontend/Admin**: Vercel (free)
- **Backend**: Render (free)
- **Database**: Neon (free PostgreSQL with pgvector)
- **Images**: Cloudinary (free)

## API Endpoints

### Main App

- `POST /api/verify` - Submit face for verification
- `POST /api/verify/dispute` - Submit dispute

### Admin

- `GET /api/admin/stats` - Dashboard statistics
- `GET /api/admin/reviews` - List pending reviews
- `POST /api/admin/reviews/:id/approve` - Approve registration
- `POST /api/admin/reviews/:id/reject` - Reject as duplicate

## Tech Stack

- **Frontend**: React, Vite, Tailwind CSS, MediaPipe
- **Backend**: FastAPI, SQLAlchemy, InsightFace
- **Database**: PostgreSQL with pgvector
- **ML Models**: MediaPipe (frontend), InsightFace ArcFace (backend)

## License

MIT
