from app.db.session import Base

# Import models so SQLAlchemy registers them
from app.models.database import (
    CustomerFace,
    ReviewQueue,
    DisputeImage,
    AuditLog
)