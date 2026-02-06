# Deployment Guide

## Overview

All services deployed on free tiers:

| Service | Provider | Free Tier |
|---------|----------|-----------|
| Frontend (main) | Vercel | Unlimited |
| Frontend (admin) | Vercel | Unlimited |
| Backend | Render | 750 hrs/month |
| Database | Neon | 0.5 GB |
| Image Storage | Cloudinary | 25 GB |

## Step 1: Database Setup (Neon)

### 1.1 Create Account
1. Go to https://neon.tech
2. Sign up with GitHub

### 1.2 Create Project
1. Click "Create Project"
2. Name: `kyc-liveness`
3. Region: Choose closest to your users
4. Click "Create"

### 1.3 Enable pgvector
1. Go to SQL Editor in dashboard
2. Run:
```sql
CREATE EXTENSION IF NOT EXISTS vector;
```

### 1.4 Create Tables
Run the full schema from `docs/DATABASE.md` in SQL Editor.

### 1.5 Get Connection String
1. Go to Dashboard → Connection Details
2. Copy the connection string
3. Format: `postgresql://user:pass@host/db?sslmode=require`

For async Python, modify to:
`postgresql+asyncpg://user:pass@host/db?ssl=require`

---

## Step 2: Image Storage (Cloudinary)

### 2.1 Create Account
1. Go to https://cloudinary.com
2. Sign up (free)

### 2.2 Get Credentials
1. Go to Dashboard
2. Note down:
   - Cloud Name
   - API Key
   - API Secret

### 2.3 Create Upload Preset (Optional)
1. Settings → Upload
2. Add upload preset
3. Name: `kyc_disputes`
4. Folder: `disputes`
5. Signing mode: Unsigned (for simplicity) or Signed

---

## Step 3: Backend Deployment (Render)

### 3.1 Prepare Repository
Ensure `backend/` folder contains:
- `requirements.txt`
- `render.yaml`
- All Python code

### 3.2 Create render.yaml

```yaml
# backend/render.yaml
services:
  - type: web
    name: kyc-api
    env: python
    region: oregon
    plan: free
    buildCommand: pip install -r requirements.txt
    startCommand: uvicorn app.main:app --host 0.0.0.0 --port $PORT
    envVars:
      - key: DATABASE_URL
        sync: false
      - key: CLOUDINARY_CLOUD_NAME
        sync: false
      - key: CLOUDINARY_API_KEY
        sync: false
      - key: CLOUDINARY_API_SECRET
        sync: false
      - key: CORS_ORIGINS
        sync: false
```

### 3.3 Deploy
1. Go to https://render.com
2. Sign up with GitHub
3. New → Web Service
4. Connect repository
5. Select `backend` folder as root directory
6. Render auto-detects from `render.yaml`
7. Add environment variables:
   - `DATABASE_URL` = Neon connection string (asyncpg version)
   - `CLOUDINARY_CLOUD_NAME` = from Cloudinary
   - `CLOUDINARY_API_KEY` = from Cloudinary
   - `CLOUDINARY_API_SECRET` = from Cloudinary
   - `CORS_ORIGINS` = `https://your-app.vercel.app,https://your-admin.vercel.app`
8. Deploy

### 3.4 Note API URL
After deploy, note the URL: `https://kyc-api-xxxx.onrender.com`

**Important:** Free tier sleeps after 15 min inactivity. First request takes ~30s.

---

## Step 4: Frontend Deployment (Vercel)

### 4.1 Main App

1. Go to https://vercel.com
2. Sign up with GitHub
3. Import repository
4. Configure:
   - Framework: Vite
   - Root Directory: `frontend`
   - Build Command: `npm run build`
   - Output Directory: `dist`
5. Environment Variables:
   - `VITE_API_URL` = `https://kyc-api-xxxx.onrender.com`
6. Deploy

### 4.2 Admin App

1. In Vercel, click "Add New Project"
2. Import same repository
3. Configure:
   - Framework: Vite
   - Root Directory: `admin`
   - Build Command: `npm run build`
   - Output Directory: `dist`
4. Environment Variables:
   - `VITE_API_URL` = `https://kyc-api-xxxx.onrender.com`
5. Deploy
6. (Optional) Add custom domain: `admin.yourdomain.com`

---

## Step 5: Domain Configuration (Optional)

### Custom Domain on Vercel

1. Go to Project Settings → Domains
2. Add domain: `app.yourdomain.com`
3. Follow DNS instructions

### CORS Update

After adding custom domains, update backend `CORS_ORIGINS`:
```
CORS_ORIGINS=https://app.yourdomain.com,https://admin.yourdomain.com
```

---

## Environment Variables Summary

### Backend (Render)

| Variable | Example |
|----------|---------|
| `DATABASE_URL` | `postgresql+asyncpg://user:pass@host/db?ssl=require` |
| `CLOUDINARY_CLOUD_NAME` | `dxxxxxx` |
| `CLOUDINARY_API_KEY` | `123456789` |
| `CLOUDINARY_API_SECRET` | `abcdefghijk` |
| `CORS_ORIGINS` | `https://app.vercel.app,https://admin.vercel.app` |

### Frontend (Vercel)

| Variable | Example |
|----------|---------|
| `VITE_API_URL` | `https://kyc-api-xxxx.onrender.com` |

### Admin (Vercel)

| Variable | Example |
|----------|---------|
| `VITE_API_URL` | `https://kyc-api-xxxx.onrender.com` |

---

## Local Development

### Prerequisites
- Node.js 18+
- Python 3.11+
- Docker (for local PostgreSQL)

### Database (Local)
```bash
# Using Docker
docker run --name kyc-postgres \
  -e POSTGRES_USER=kyc \
  -e POSTGRES_PASSWORD=localdev \
  -e POSTGRES_DB=kyc_db \
  -p 5432:5432 \
  -d pgvector/pgvector:pg16

# Connect and setup
psql postgresql://kyc:localdev@localhost:5432/kyc_db
# Run schema from docs/DATABASE.md
```

### Backend (Local)
```bash
cd backend
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows
pip install -r requirements.txt

# Set environment variables
export DATABASE_URL="postgresql+asyncpg://kyc:localdev@localhost:5432/kyc_db"
export CORS_ORIGINS="http://localhost:5173,http://localhost:5174"

# Run
uvicorn app.main:app --reload --port 8000
```

### Frontend (Local)
```bash
cd frontend
npm install

# Create .env.local
echo "VITE_API_URL=http://localhost:8000" > .env.local

npm run dev  # Runs on http://localhost:5173
```

### Admin (Local)
```bash
cd admin
npm install

# Create .env.local
echo "VITE_API_URL=http://localhost:8000" > .env.local

npm run dev -- --port 5174  # Runs on http://localhost:5174
```

---

## Troubleshooting

### Backend won't start on Render

**Check logs for:**
- Missing dependencies → Add to `requirements.txt`
- Model download failing → Models may need manual setup
- Memory issues → Free tier has 512MB limit

**Solution for models:**
Include model files in repo or use smaller models.

### CORS errors

**Symptoms:** Browser console shows CORS blocked

**Fix:**
1. Ensure `CORS_ORIGINS` includes exact frontend URL
2. No trailing slash
3. Include protocol (`https://`)

### Database connection fails

**Check:**
1. Connection string format (asyncpg requires `postgresql+asyncpg://`)
2. SSL mode (`?ssl=require` for Neon)
3. IP allowlist (Neon allows all by default)

### Vercel build fails

**Check:**
1. Root directory is set correctly (`frontend` or `admin`)
2. `package.json` exists in root directory
3. Build command matches (`npm run build`)

### First request is slow

**Expected behavior** on free tiers:
- Render: ~30s cold start after 15 min idle
- Neon: ~1-2s cold start after idle

**Workarounds:**
- Use health check endpoint to keep warm
- Accept it for hackathon demo (warn judges)

---

## Monitoring

### Render
- Dashboard shows request logs
- Set up alerts for errors

### Vercel
- Analytics tab shows traffic
- Functions tab shows API routes (if using)

### Neon
- Dashboard shows query stats
- Monitor storage usage (0.5GB limit)

---

## Checklist Before Demo

- [ ] All services deployed and accessible
- [ ] CORS configured correctly
- [ ] Database tables created
- [ ] Test full flow: camera → challenges → verify → result
- [ ] Test admin review flow
- [ ] Test duplicate detection
- [ ] Have backup plan if Render cold starts during demo
