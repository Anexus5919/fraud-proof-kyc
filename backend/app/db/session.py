import ssl
import re
from urllib.parse import urlparse, parse_qs, unquote
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
from sqlalchemy.orm import declarative_base

from app.config import get_settings

settings = get_settings()

# Parse database URL and handle SSL/options for asyncpg
database_url = settings.database_url

# Extract endpoint from options parameter if present
endpoint_id = None
if "options=" in database_url:
    match = re.search(r'options=endpoint%3D([^&]+)', database_url)
    if match:
        endpoint_id = unquote(match.group(1))

# Remove parameters asyncpg doesn't support as URL params
# (sslmode and options - we'll handle them via connect_args)
database_url = re.sub(r'[?&]sslmode=[^&]*', '', database_url)
database_url = re.sub(r'[?&]options=[^&]*', '', database_url)
# Clean up URL
database_url = database_url.rstrip('?&')
if '?' not in database_url and '&' in database_url:
    database_url = database_url.replace('&', '?', 1)

# Create SSL context for Neon (requires SSL)
ssl_context = ssl.create_default_context()
ssl_context.check_hostname = False
ssl_context.verify_mode = ssl.CERT_NONE

# Build connect_args for Neon
connect_args = {}
if "neon.tech" in settings.database_url:
    connect_args["ssl"] = ssl_context
    # Pass endpoint ID via server_settings for SNI workaround
    if endpoint_id:
        connect_args["server_settings"] = {"options": f"endpoint={endpoint_id}"}

# Create async engine with SSL and robust connection pooling
engine = create_async_engine(
    database_url,
    echo=False,
    future=True,
    connect_args=connect_args,
    # Connection pool settings for reliability
    pool_pre_ping=True,  # Check connection health before use
    pool_size=5,  # Number of connections to keep open
    max_overflow=10,  # Extra connections when pool is exhausted
    pool_recycle=300,  # Recycle connections after 5 minutes
    pool_timeout=30,  # Wait up to 30s for a connection
)

# Create async session factory
async_session = async_sessionmaker(
    engine,
    class_=AsyncSession,
    expire_on_commit=False,
)

# Base class for models
Base = declarative_base()


# Dependency for FastAPI
async def get_db():
    async with async_session() as session:
        try:
            yield session
        finally:
            await session.close()
