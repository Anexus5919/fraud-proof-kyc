import logging
from pathlib import Path
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.config import get_settings
from app.routers import verify, admin

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

settings = get_settings()

# Startup config validation
_env_path = Path(__file__).resolve().parent.parent / ".env"
logger.info(f"[STARTUP] .env file path: {_env_path}")
logger.info(f"[STARTUP] .env file exists: {_env_path.exists()}")
_db_url = settings.database_url
_masked = _db_url.split("@")[0].rsplit(":", 1)[0] + ":****@" + _db_url.split("@")[1] if "@" in _db_url else _db_url
logger.info(f"[STARTUP] DATABASE_URL loaded: {_masked}")
logger.info(f"[STARTUP] CORS origins: {settings.cors_origins}")
logger.info(f"[STARTUP] Spoof threshold: {settings.spoof_threshold}")
logger.info(f"[STARTUP] Duplicate threshold: {settings.duplicate_threshold}")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan handler"""
    # Startup
    logger.info("Starting KYC Liveness Detection API...")

    # Pre-load ML models for faster first request
    try:
        from app.services.face_embedder import get_face_app
        from app.services.spoof_detector import get_spoof_detector

        logger.info("Loading ML models...")
        get_face_app()  # Load InsightFace
        get_spoof_detector()  # Load spoof detector
        logger.info("ML models loaded successfully")
    except Exception as e:
        logger.error(f"Failed to pre-load ML models: {e}")

    yield

    # Shutdown
    logger.info("Shutting down...")


# Create FastAPI app
app = FastAPI(
    title="KYC Liveness Detection API",
    description="API for facial liveness detection and deduplication",
    version="1.0.0",
    lifespan=lifespan
)

# Configure CORS
origins = [origin.strip() for origin in settings.cors_origins.split(",") if origin.strip()]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Request/Response logging middleware
import time as _time
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request as StarletteRequest


class RequestLoggingMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: StarletteRequest, call_next):
        start = _time.time()
        logger.info(f"[HTTP] --> {request.method} {request.url.path} (from {request.client.host if request.client else '?'})")
        response = await call_next(request)
        elapsed = _time.time() - start
        logger.info(f"[HTTP] <-- {request.method} {request.url.path} → {response.status_code} ({elapsed:.3f}s)")
        return response


app.add_middleware(RequestLoggingMiddleware)

# Include routers
app.include_router(verify.router)
app.include_router(admin.router)


@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "service": "KYC Liveness Detection API",
        "version": "1.0.0"
    }


@app.get("/health")
async def health():
    """Detailed health check"""
    return {
        "status": "healthy",
        "database": "connected",  # TODO: Add actual DB check
        "ml_models": "loaded"  # TODO: Add actual model check
    }
