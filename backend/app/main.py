import logging
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
origins = [origin.strip() for origin in settings.cors_origins.split(",")]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

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
