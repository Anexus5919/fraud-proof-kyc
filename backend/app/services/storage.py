import base64
import logging
from typing import Optional
import cloudinary
import cloudinary.uploader

from app.config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()

# Initialize Cloudinary
_cloudinary_initialized = False


def init_cloudinary():
    """Initialize Cloudinary with credentials"""
    global _cloudinary_initialized

    if _cloudinary_initialized:
        return

    if settings.cloudinary_cloud_name and settings.cloudinary_api_key:
        cloudinary.config(
            cloud_name=settings.cloudinary_cloud_name,
            api_key=settings.cloudinary_api_key,
            api_secret=settings.cloudinary_api_secret
        )
        _cloudinary_initialized = True
        logger.info("Cloudinary initialized")
    else:
        logger.warning("Cloudinary credentials not configured, image upload disabled")


def upload_image(
    image_base64: str,
    folder: str = "kyc",
    public_id: Optional[str] = None
) -> Optional[str]:
    """
    Upload a base64 encoded image to Cloudinary.

    Args:
        image_base64: Base64 encoded image data
        folder: Cloudinary folder name
        public_id: Optional public ID for the image

    Returns:
        URL of uploaded image or None if upload failed
    """
    init_cloudinary()

    if not _cloudinary_initialized:
        logger.warning("Cloudinary not initialized, skipping upload")
        return None

    try:
        # Prepare data URL
        data_url = f"data:image/jpeg;base64,{image_base64}"

        # Upload options
        options = {
            "folder": folder,
            "resource_type": "image",
        }

        if public_id:
            options["public_id"] = public_id

        # Upload to Cloudinary
        result = cloudinary.uploader.upload(data_url, **options)

        return result.get("secure_url")

    except Exception as e:
        logger.error(f"Failed to upload image to Cloudinary: {e}")
        return None


def delete_image(public_id: str) -> bool:
    """
    Delete an image from Cloudinary.

    Args:
        public_id: Public ID of the image to delete

    Returns:
        True if deleted successfully
    """
    init_cloudinary()

    if not _cloudinary_initialized:
        return False

    try:
        result = cloudinary.uploader.destroy(public_id)
        return result.get("result") == "ok"
    except Exception as e:
        logger.error(f"Failed to delete image from Cloudinary: {e}")
        return False
