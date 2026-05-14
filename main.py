"""
╔══════════════════════════════════════════════════════════╗
║  Skin Cancer Detection API — FastAPI + TFLite Backend   ║
║  File: backend/main.py                                  ║
╚══════════════════════════════════════════════════════════╝
"""

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import numpy as np
from PIL import Image
import tensorflow as tf 
tflite = tf.lite
# If tflite_runtime is not available, use: import tensorflow as tf
# and replace tflite.Interpreter with tf.lite.Interpreter
import io
import os
import time
import uuid
from datetime import datetime
from pydantic import BaseModel

# ── App init ──────────────────────────────────────────────
app = FastAPI(
    title="SkinScan AI API",
    description="Skin cancer detection using TFLite model",
    version="1.0.0"
)

# ── CORS — allows your frontend (any origin) to call this API ──
# In production, replace "*" with your actual frontend URL
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── TFLite Model Loading ───────────────────────────────────
MODEL_PATH = os.path.join(os.path.dirname(__file__), "tiny_model.tflite")

def load_model():
    """Load TFLite interpreter once at startup."""
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"Model not found at {MODEL_PATH}. "
            "Place your tiny_model.tflite in the backend/ folder."
        )
    interpreter = tflite.Interpreter(model_path=MODEL_PATH)
    interpreter.allocate_tensors()
    return interpreter

# Load model at startup (singleton pattern)
try:
    interpreter = load_model()
    input_details  = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # Auto-detect input shape from model (e.g. [1, 224, 224, 3])
    INPUT_SHAPE = input_details[0]['shape']
    IMG_HEIGHT  = INPUT_SHAPE[1]
    IMG_WIDTH   = INPUT_SHAPE[2]
    print(f"✅  Model loaded — input shape: {INPUT_SHAPE}")
except Exception as e:
    print(f"⚠️  Model load failed: {e}")
    interpreter = None


# ── Preprocessing ──────────────────────────────────────────
def preprocess_image(image_bytes: bytes) -> np.ndarray:
    """
    Convert raw image bytes → model-ready numpy array.
    Steps: Open → RGB → Resize → Normalize [0,1] → Add batch dim
    """
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    img = img.resize((IMG_WIDTH, IMG_HEIGHT))
    arr = np.array(img, dtype=np.float32) / 255.0   # normalize to [0, 1]
    arr = np.expand_dims(arr, axis=0)                # (H,W,3) → (1,H,W,3)
    return arr


# ── Inference ──────────────────────────────────────────────
def run_inference(image_array: np.ndarray) -> float:
    """
    Run TFLite inference and return probability of malignancy.
    """
    interpreter.set_tensor(input_details[0]['index'], image_array)
    interpreter.invoke()
    output = interpreter.get_tensor(output_details[0]['index'])
    # output[0][0] → sigmoid probability (0=Benign, 1=Malignant)
    return float(output[0][0])


# ── Risk Stratification ────────────────────────────────────
def get_risk_level(prob: float) -> dict:
    """Map probability to risk level, label, and clinical action."""
    if prob >= 0.75:
        return {
            "level": "HIGH",
            "color": "#ef4444",
            "badge": "🔴 HIGH RISK",
            "action": "Urgent dermatologist referral recommended. Biopsy may be required.",
            "description": "Strong malignant features detected. Immediate clinical evaluation is advised."
        }
    elif prob >= 0.45:
        return {
            "level": "MEDIUM",
            "color": "#f59e0b",
            "badge": "🟡 MODERATE RISK",
            "action": "Schedule a dermatologist appointment within 2–4 weeks.",
            "description": "Ambiguous features present. Professional clinical assessment is recommended."
        }
    else:
        return {
            "level": "LOW",
            "color": "#10b981",
            "badge": "🟢 LOW RISK",
            "action": "Continue routine annual skin checks.",
            "description": "Features appear predominantly benign. Monitor for any changes."
        }


# ── Routes ─────────────────────────────────────────────────

@app.get("/")
def root():
    return {"message": "SkinScan AI API is running", "status": "ok"}


@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "model_loaded": interpreter is not None,
        "model_input_shape": INPUT_SHAPE.tolist() if interpreter else None,
        "timestamp": datetime.now().isoformat()
    }


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """
    Main prediction endpoint.
    Accepts: multipart/form-data with an image file
    Returns: JSON diagnosis report
    """
    if interpreter is None:
        raise HTTPException(status_code=503, detail="Model not loaded. Check server logs.")

    # ── Validate file type ────────────────────────────────
    allowed_types = ["image/jpeg", "image/png", "image/jpg", "image/webp"]
    if file.content_type not in allowed_types:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid file type '{file.content_type}'. Upload JPEG or PNG."
        )

    # ── Validate file size (max 10MB) ─────────────────────
    contents = await file.read()
    if len(contents) > 10 * 1024 * 1024:
        raise HTTPException(status_code=400, detail="File too large. Max 10MB.")

    try:
        # ── Preprocess & infer ────────────────────────────
        start_time   = time.time()
        image_array  = preprocess_image(contents)
        prob_mal     = run_inference(image_array)
        latency_ms   = round((time.time() - start_time) * 1000, 2)

        prob_benign  = 1.0 - prob_mal
        prediction   = "Malignant" if prob_mal >= 0.5 else "Benign"
        confidence   = prob_mal if prediction == "Malignant" else prob_benign
        risk         = get_risk_level(prob_mal)

        return JSONResponse({
            "scan_id"          : str(uuid.uuid4())[:8].upper(),
            "timestamp"        : datetime.now().isoformat(),
            "filename"         : file.filename,
            "prediction"       : prediction,
            "confidence"       : round(confidence * 100, 2),
            "p_malignant"      : round(prob_mal * 100, 2),
            "p_benign"         : round(prob_benign * 100, 2),
            "risk_level"       : risk["level"],
            "risk_color"       : risk["color"],
            "risk_badge"       : risk["badge"],
            "risk_description" : risk["description"],
            "recommended_action": risk["action"],
            "inference_ms"     : latency_ms,
            "model_input_size" : f"{IMG_WIDTH}×{IMG_HEIGHT}",
            "disclaimer": (
                "⚠️ For research/educational use only. "
                "Not a substitute for professional medical diagnosis."
            )
        })

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference failed: {str(e)}")


@app.get("/stats")
def get_stats():
    """Mock endpoint for dashboard statistics."""
    return {
        "total_scans": 1284,
        "malignant_detected": 187,
        "benign_detected": 1097,
        "accuracy": 94.3,
        "model_version": "EfficientNetB0-TFLite-v1",
        "last_updated": "2026-05-10"
    }
