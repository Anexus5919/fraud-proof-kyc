from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from datetime import datetime
from uuid import UUID


# Motion analysis data from client-side tracking
class MotionAnalysis(BaseModel):
    frames_analyzed: Optional[int] = Field(None, alias="framesAnalyzed")
    micro_movement_avg: Optional[float] = Field(None, alias="microMovementAvg")
    parallel_motion_corr: Optional[float] = Field(None, alias="parallelMotionCorr")
    motion_variance_spread: Optional[float] = Field(None, alias="motionVarianceSpread")
    liveness_score: Optional[float] = Field(None, alias="livenessScore")
    flags: Optional[List[str]] = None

    class Config:
        populate_by_name = True


# Verification request/response
class VerifyRequest(BaseModel):
    image: str = Field(..., description="Base64 encoded image")
    session_id: str = Field(..., description="Unique session ID")
    challenges_completed: List[str] = Field(default=[], description="List of completed challenge IDs")
    motion_analysis: Optional[MotionAnalysis] = Field(None, alias="motionAnalysis", description="Client-side motion analysis data")

    class Config:
        populate_by_name = True


class VerifyResponse(BaseModel):
    """
    5-Layer Verification Response

    Statuses:
    - success: Auto approved (risk 0-30)
    - pending_review: Sent to manual review (risk 31-60)
    - rejected: Auto rejected (risk 61-100)
    - spoof_detected: Layer 2 failure
    - deepfake_detected: Layer 3 failure
    - duplicate_found: Layer 4 flagged
    - error: Processing error
    """
    status: str = Field(..., description="Verification result status")
    message: Optional[str] = None
    customer_id: Optional[str] = None
    confidence: Optional[float] = None
    can_dispute: bool = False
    review_id: Optional[str] = None

    # 5-Layer System Results
    risk_score: Optional[int] = Field(None, description="Risk score 0-100")
    risk_level: Optional[str] = Field(None, description="Risk level: low, medium, high")
    layer_results: Optional[Dict[str, Any]] = Field(None, description="Individual layer results")


# Dispute request/response
class DisputeRequest(BaseModel):
    session_id: str
    image: str = Field(..., description="Base64 encoded image")
    reason: str


class DisputeResponse(BaseModel):
    status: str
    message: str


# Admin review schemas
class ReviewItem(BaseModel):
    id: UUID
    new_customer_id: str
    new_customer_name: Optional[str]
    new_face_image_url: Optional[str]
    matched_customer_id: Optional[str]
    matched_customer_name: Optional[str]
    matched_face_image_url: Optional[str]
    similarity_score: float
    status: str
    created_at: datetime

    class Config:
        from_attributes = True


class ReviewListResponse(BaseModel):
    reviews: List[ReviewItem]
    total: int


class ReviewActionRequest(BaseModel):
    notes: Optional[str] = None


class ReviewActionResponse(BaseModel):
    status: str
    message: str


# Dispute review schemas
class DisputeItem(BaseModel):
    id: UUID
    session_id: str
    image_url: str
    reason: str
    status: str
    created_at: datetime

    class Config:
        from_attributes = True


class DisputeListResponse(BaseModel):
    disputes: List[DisputeItem]
    total: int
