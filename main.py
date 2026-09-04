"""
main.py
-------
The FastAPI application. This is the file you run to start the
backend server.

Endpoints:
  GET  /         -> simple status message
  GET  /health   -> {"status": "ok"}
  POST /analyze  -> accepts an audio file, returns prediction JSON
"""

import os
import uuid
import traceback

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from audio_processor import extract_features
from model import predict_synthetic_probability, is_demo_mode
from risk import calculate_risk_score, get_risk_level, get_security_message

app = FastAPI(title="NexGen AI Voice Security")

# ---------------------------------------------------------------
# CORS: allows the frontend (opened as a local HTML file or via a
# simple dev server) to call this API from the browser.
# For a hackathon MVP we allow all origins to keep things simple.
# ---------------------------------------------------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---- Upload validation settings ----
ALLOWED_EXTENSIONS = {".wav", ".mp3", ".m4a"}
MAX_FILE_SIZE_BYTES = 15 * 1024 * 1024  # 15 MB

# Temp files are stored in this folder and deleted after analysis
TEMP_DIR = os.path.join(os.path.dirname(__file__), "temp_uploads")
os.makedirs(TEMP_DIR, exist_ok=True)


@app.get("/")
def read_root():
    """Simple health/status message for the root URL."""
    return {"message": "NexGen AI Voice Security backend is running."}


@app.get("/health")
def health_check():
    """Used to confirm the server is alive."""
    return {"status": "ok"}


@app.post("/analyze")
async def analyze_audio(file: UploadFile = File(...)):
    """
    Main endpoint. Accepts an uploaded audio file and returns a
    prediction, confidence, risk score, risk level, and message.
    """
    temp_file_path = None
    try:
        # ---- 1. Validate file type ----
        original_name = file.filename or ""
        _, extension = os.path.splitext(original_name.lower())

        if extension not in ALLOWED_EXTENSIONS:
            raise HTTPException(
                status_code=400,
                detail="Unsupported file type. Please upload a WAV, MP3, or M4A file.",
            )

        # ---- 2. Read file bytes and validate size ----
        file_bytes = await file.read()
        if len(file_bytes) == 0:
            raise HTTPException(status_code=400, detail="Uploaded file is empty.")

        if len(file_bytes) > MAX_FILE_SIZE_BYTES:
            raise HTTPException(
                status_code=400,
                detail="File is too large. Please upload a file under 15MB.",
            )

        # ---- 3. Save it temporarily with a safe, random filename ----
        safe_filename = f"{uuid.uuid4().hex}{extension}"
        temp_file_path = os.path.join(TEMP_DIR, safe_filename)
        with open(temp_file_path, "wb") as temp_file:
            temp_file.write(file_bytes)

        # ---- 4-8. Load audio, extract features, run model ----
        # (extract_features is called inside predict_synthetic_probability,
        # keeping training/prediction feature logic in one shared place)
        synthetic_probability = predict_synthetic_probability(temp_file_path)

        # ---- 9-10. Calculate risk score and level ----
        risk_score = calculate_risk_score(synthetic_probability)
        risk_level = get_risk_level(risk_score)
        security_message = get_security_message(risk_level)

        prediction = "SUSPICIOUS" if synthetic_probability >= 0.5 else "GENUINE"

        # Confidence = how far the probability is from the 50/50 line,
        # expressed as a percentage. E.g. 0.91 synthetic -> 91% confidence
        # it's synthetic. 0.10 synthetic -> 90% confidence it's genuine.
        confidence = synthetic_probability if prediction == "SUSPICIOUS" else (1 - synthetic_probability)
        confidence_percent = round(confidence * 100, 1)

        explanation = (
            "AI analysis indicates characteristics consistent with a "
            "potentially synthetic voice."
            if prediction == "SUSPICIOUS"
            else "AI analysis indicates characteristics more consistent "
            "with a natural human voice."
        )

        response = {
            "prediction": prediction,
            "confidence": confidence_percent,
            "risk_score": risk_score,
            "risk_level": risk_level,
            "message": security_message,
            "explanation": explanation,
            "demo_mode": is_demo_mode(),
        }

        # ---- 11. Return JSON ----
        return JSONResponse(content=response)

    except HTTPException:
        # Re-raise HTTPExceptions as-is (these already have clean messages)
        raise
    except Exception:
        # Never leak a Python stack trace to the user.
        # Log the real error server-side for debugging.
        print("Unexpected error during analysis:")
        print(traceback.format_exc())
        raise HTTPException(
            status_code=500,
            detail="Unable to analyze this audio. Please upload a valid WAV, MP3, or M4A file.",
        )
    finally:
        # ---- 12. Delete the temporary file no matter what happened ----
        if temp_file_path and os.path.isfile(temp_file_path):
            os.remove(temp_file_path)
