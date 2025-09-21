"""
OMR Grader API - FastAPI Backend

This API provides endpoints for grading Optical Mark Recognition (OMR) sheets.
Students can upload their answer sheets and receive automated grading results.

Author: Giri Krishna
Created: September 2025
"""

import os
import uuid
import cv2
import numpy as np
import traceback
from typing import List, Dict, Optional

from fastapi import FastAPI, UploadFile, File, Query, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from .omr_core import grade_from_bytes
from .models import init_db, SessionLocal, Submission, Detail

# Initialize the FastAPI application
app = FastAPI(
    title="OMR Grader API",
    description="Automated grading system for OMR answer sheets",
    version="1.0.0",
    contact={
        "name": "Support Team",
        "email": "support@omrgrader.com"
    }
)

# Configure CORS to allow frontend access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify exact origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Set up static file serving for overlay images
BASE_DIR = os.path.dirname(__file__)
STATIC_DIR = os.path.join(BASE_DIR, "static")
OVERLAY_DIR = os.path.join(STATIC_DIR, "overlays")

# Ensure directories exist
os.makedirs(OVERLAY_DIR, exist_ok=True)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

# Initialize database
init_db()

# ==================== Data Models ====================

class GradeResponse(BaseModel):
    """Response model for grading results"""
    id: int = Field(..., description="Unique submission ID")
    total_correct: int = Field(..., description="Total number of correct answers")
    subject_scores: Dict[str, int] = Field(..., description="Scores breakdown by subject")
    overlay_url: str = Field(..., description="URL to the graded overlay image")
    
    class Config:
        schema_extra = {
            "example": {
                "id": 1,
                "total_correct": 85,
                "subject_scores": {
                    "Python": 18,
                    "Data Analysis": 17,
                    "MySQL": 16,
                    "Power BI": 19,
                    "Adv Stats": 15
                },
                "overlay_url": "/static/overlays/abc123.png"
            }
        }

class SubmissionSummary(BaseModel):
    """Summary model for listing submissions"""
    id: int = Field(..., description="Submission ID")
    image_name: str = Field(..., description="Original image filename")
    total_correct: int = Field(..., description="Total correct answers")
    python_score: int = Field(..., description="Python subject score")
    data_analysis_score: int = Field(..., description="Data Analysis score")
    mysql_score: int = Field(..., description="MySQL score")
    power_bi_score: int = Field(..., description="Power BI score")
    adv_stats_score: int = Field(..., description="Advanced Stats score")
    overlay_url: str = Field(..., description="Graded image URL")
    created_at: Optional[str] = Field(None, description="Submission timestamp")

class QuestionDetail(BaseModel):
    """Individual question response detail"""
    subject: str = Field(..., description="Subject name")
    question_number: int = Field(..., description="Question number")
    predicted_answer: str = Field(..., description="Detected answer choice")
    
    class Config:
        schema_extra = {
            "example": {
                "subject": "Python",
                "question_number": 1,
                "predicted_answer": "A"
            }
        }

# ==================== API Endpoints ====================

@app.get("/health")
def health_check():
    """
    Health check endpoint to verify the API is running.
    
    Returns a simple status message to confirm the service is operational.
    Useful for load balancers and monitoring systems.
    """
    return {
        "status": "healthy", 
        "message": "OMR Grader API is running successfully",
        "version": "1.0.0"
    }

def save_overlay_image(overlay_data, output_path: str) -> None:
    """
    Intelligently save overlay data as a PNG image.
    
    This function handles various input formats including:
    - NumPy arrays (most common)
    - Raw bytes data
    - Base64 encoded strings
    - File paths to existing images
    
    Args:
        overlay_data: The overlay data in various supported formats
        output_path: Where to save the final PNG file
        
    Raises:
        HTTPException: If the data cannot be processed or saved
    """
    import base64
    import re
    import shutil

    def save_numpy_array(img_array: np.ndarray):
        """Helper function to save a NumPy array as an image"""
        if img_array is None:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Image data is empty - cannot save overlay"
            )
            
        # Validate image dimensions
        if img_array.ndim == 2:  # Grayscale image
            pass
        elif img_array.ndim == 3 and img_array.shape[2] in (3, 4):  # Color image (BGR/BGRA)
            pass
        else:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid image dimensions: {img_array.shape}. Expected 2D or 3D array."
            )
        
        # Ensure proper data type for image saving
        if img_array.dtype != np.uint8:
            img_array = np.clip(img_array, 0, 255).astype(np.uint8)
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Try to save the image
        success = cv2.imwrite(output_path, img_array)
        if not success:
            # Fallback: encode as PNG buffer and write manually
            encode_success, buffer = cv2.imencode(".png", img_array)
            if not encode_success:
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail="Failed to encode overlay image"
                )
            with open(output_path, "wb") as file:
                file.write(buffer.tobytes())

    # Handle NumPy array input (most common case)
    if isinstance(overlay_data, np.ndarray):
        save_numpy_array(overlay_data)
        return

    # Handle raw bytes input
    if isinstance(overlay_data, (bytes, bytearray)):
        img_array = np.frombuffer(overlay_data, np.uint8)
        decoded_img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        save_numpy_array(decoded_img)
        return

    # Handle string input (file path or base64)
    if isinstance(overlay_data, str):
        data_string = overlay_data.strip()
        
        # Check if it's a file path
        if os.path.exists(data_string):
            existing_img = cv2.imread(data_string, cv2.IMREAD_UNCHANGED)
            if existing_img is not None:
                save_numpy_array(existing_img)
                return
            # If cv2 can't read it, try copying the file directly
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            shutil.copy(data_string, output_path)
            return
        
        # Try to decode as base64
        try:
            # Handle data URLs (e.g., "data:image/png;base64,...")
            base64_match = re.match(r"^data:image/[^;]+;base64,(.+)$", data_string, flags=re.I)
            base64_data = base64_match.group(1) if base64_match else data_string
            
            decoded_bytes = base64.b64decode(base64_data, validate=True)
            img_array = np.frombuffer(decoded_bytes, np.uint8)
            decoded_img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
            save_numpy_array(decoded_img)
            return
        except Exception:
            pass
            
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="String input is neither a valid file path nor valid base64 image data"
        )

    # If none of the above formats match
    raise HTTPException(
        status_code=status.HTTP_400_BAD_REQUEST,
        detail=f"Unsupported overlay data type: {type(overlay_data)}. Expected NumPy array, bytes, or string."
    )

@app.post("/grade", response_model=GradeResponse)
async def grade_omr_sheet(
    image: UploadFile = File(..., description="OMR answer sheet image (JPEG/PNG)"),
    key: UploadFile = File(..., description="Answer key file (Excel/CSV)")
):
    """
    Grade an OMR answer sheet and return the results.
    
    This endpoint processes uploaded OMR sheets by:
    1. Reading the student's marked answers
    2. Comparing against the provided answer key
    3. Calculating scores for each subject
    4. Generating a visual overlay showing detected marks
    5. Storing results in the database
    
    Args:
        image: The scanned OMR answer sheet
        key: Answer key in Excel or CSV format
        
    Returns:
        GradeResponse: Complete grading results with scores and overlay
    """
    try:
        # Read uploaded files
        student_sheet = await image.read()
        answer_key = await key.read()

        # Process the OMR sheet and generate results
        grade_summary, question_details, overlay_image = grade_from_bytes(student_sheet, answer_key)

        # Save the overlay image with a unique filename
        overlay_filename = f"graded_{uuid.uuid4().hex}.png"
        overlay_path = os.path.join(OVERLAY_DIR, overlay_filename)
        save_overlay_image(overlay_image, overlay_path)
        overlay_url = f"/static/overlays/{overlay_filename}"

        # Store results in database
        db = SessionLocal()
        try:
            # Create new submission record
            new_submission = Submission(
                image_name=image.filename or "unknown.jpg",
                overlay_path=overlay_url,
                total_correct=grade_summary.get("total_correct", 0),
                python_score=grade_summary.get("Python_score", 0),
                data_analysis_score=grade_summary.get("Data Analysis_score", 0),
                mysql_score=grade_summary.get("MySQL_score", 0),
                power_bi_score=grade_summary.get("Power BI_score", 0),
                adv_stats_score=grade_summary.get("Adv Stats_score", 0),
            )
            
            db.add(new_submission)
            db.flush()  # Get the submission ID
            
            # Store individual question details
            for detail in question_details:
                question_detail = Detail(
                    submission_id=new_submission.id,
                    subject=detail["Subject"],
                    qno=detail["QNo"],
                    pred=detail["Pred"]
                )
                db.add(question_detail)
            
            db.commit()
            
        finally:
            db.close()

        # Return comprehensive results
        return GradeResponse(
            id=new_submission.id,
            total_correct=new_submission.total_correct,
            subject_scores={
                "Python": new_submission.python_score,
                "Data Analysis": new_submission.data_analysis_score,
                "MySQL": new_submission.mysql_score,
                "Power BI": new_submission.power_bi_score,
                "Adv Stats": new_submission.adv_stats_score,
            },
            overlay_url=overlay_url,
        )
        
    except HTTPException:
        # Re-raise HTTP exceptions (these have user-friendly messages)
        raise
    except Exception as e:
        # Log the full error for debugging
        traceback.print_exc()
        # Return a user-friendly error message
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Failed to process OMR sheet: {str(e)}"
        )

@app.get("/submissions", response_model=List[SubmissionSummary])
def list_submissions(limit: int = Query(50, ge=1, le=500)):
    """
    Retrieve a list of recent submissions with their scores.
    
    This endpoint returns a paginated list of OMR submissions,
    showing summary information for each graded answer sheet.
    """
    db = SessionLocal()
    try:
        submissions = db.query(Submission).order_by(Submission.id.desc()).limit(limit).all()
        return [
            SubmissionSummary(
                id=submission.id,
                image_name=submission.image_name,
                total_correct=submission.total_correct,
                python_score=submission.python_score,
                data_analysis_score=submission.data_analysis_score,
                mysql_score=submission.mysql_score,
                power_bi_score=submission.power_bi_score,
                adv_stats_score=submission.adv_stats_score,
                overlay_url=submission.overlay_path,
                created_at=submission.created_at.isoformat() if hasattr(submission, 'created_at') and submission.created_at else None
            ) for submission in submissions
        ]
    finally:
        db.close()

@app.get("/submissions/{submission_id}/details", response_model=List[QuestionDetail])
def get_submission_details(submission_id: int):
    """
    Get detailed question-by-question results for a specific submission.
    
    This endpoint returns the individual question responses detected
    for a particular OMR sheet submission, organized by subject.
    
    Args:
        submission_id: The unique ID of the submission to retrieve
        
    Returns:
        List of question details showing detected answers for each question
    """
    db = SessionLocal()
    try:
        # Fetch all question details for this submission
        question_details = db.query(Detail).filter(Detail.submission_id == submission_id).all()
        
        if not question_details:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"No submission found with ID {submission_id}"
            )
        
        # Format the response
        return [
            QuestionDetail(
                subject=detail.subject,
                question_number=detail.qno,
                predicted_answer=detail.pred or "No answer detected"
            ) 
            for detail in question_details
        ]
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve submission details: {str(e)}"
        )
    finally:
        db.close()

# ==================== Application Info ====================

@app.get("/")
def root():
    """
    Welcome endpoint with basic API information.
    """
    return {
        "message": "Welcome to the OMR Grader API",
        "description": "Automated grading system for optical mark recognition sheets",
        "documentation": "/docs",
        "health_check": "/health",
        "version": "1.0.0"
    }
