"""FastAPI application for InsightX research demonstration using ONNX Runtime.

Zero-TensorFlow serving architecture:
- ONNX Runtime CPU inference for fast, lightweight chest X-ray predictions.
- Pure NumPy Class Activation Mapping (Grad-CAM) in milliseconds.
- Platt scaling probability calibration loaded from models/calibration.json.
- Strictly in-memory image processing (Rule 4 compliant).
- Medical research disclaimer enforced across all endpoints (Rule 1 compliant).
- Firebase ID token authentication via firebase-admin with verified email check.
- Service account credentials read strictly from FIREBASE_SERVICE_ACCOUNT_JSON env var.
- Rate limiting: 10 req/min per authenticated user, 10 req/min per guest IP for samples.
"""

import base64
import io
import json
import os
import threading
import time
from collections import defaultdict
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import firebase_admin
import matplotlib.pyplot as plt
import numpy as np
import onnxruntime as ort
from fastapi import Depends, FastAPI, File, Header, HTTPException, Request, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from firebase_admin import auth, credentials
from PIL import Image, UnidentifiedImageError
from pydantic import BaseModel, Field

from insightx import RESEARCH_DISCLAIMER

# Maximum allowed upload file size: 8 MB
MAX_UPLOAD_SIZE_BYTES = 8 * 1024 * 1024  # 8,388,608 bytes
ALLOWED_CONTENT_TYPES = {"image/png", "image/jpeg", "image/jpg"}
ALLOWED_IMAGE_FORMATS = {"PNG", "JPEG"}

# Rate limit configuration
RATE_LIMIT_WINDOW_SECONDS = 60.0
USER_RATE_LIMIT_MAX = 10
GUEST_RATE_LIMIT_MAX = 10

_rate_limit_lock = threading.Lock()
_user_request_timestamps: dict[str, list[float]] = defaultdict(list)
_ip_request_timestamps: dict[str, list[float]] = defaultdict(list)


def initialize_firebase_admin() -> None:
    """Initialize Firebase Admin SDK using FIREBASE_SERVICE_ACCOUNT_JSON environment variable.

    Strictly reads credentials from the environment variable (JSON string),
    never from a file in the repository (Rule 2 & Rule 7 compliant).
    """
    if firebase_admin._apps:
        return

    sa_json_str = os.getenv("FIREBASE_SERVICE_ACCOUNT_JSON")
    if sa_json_str and sa_json_str.strip():
        try:
            sa_info = json.loads(sa_json_str)
            cred = credentials.Certificate(sa_info)
            firebase_admin.initialize_app(cred)
            print("[Firebase] Admin SDK initialized successfully from environment variable.")
        except Exception as e:
            print(f"[Firebase Warning] Failed to parse FIREBASE_SERVICE_ACCOUNT_JSON: {e}")
            firebase_admin.initialize_app()
    else:
        # Default initialization (e.g. for testing / emulators / GCP metadata)
        try:
            firebase_admin.initialize_app()
            print("[Firebase] Admin SDK initialized with default application credentials.")
        except Exception:
            pass


async def verify_firebase_bearer_token(
    authorization: str | None = Header(
        None, description="Bearer <token> Firebase ID token"
    ),
) -> dict[str, Any]:
    """Verify Firebase ID token from Authorization header and enforce verified email.

    Requires:
    1. Valid Bearer token structure.
    2. Token decodable and unexpired by Firebase Auth.
    3. User email verified (email_verified == True).
    """
    if not authorization:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing Authorization header with Bearer token.",
        )

    parts = authorization.split()
    if len(parts) != 2 or parts[0].lower() != "bearer":
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid Authorization header format. Expected 'Bearer <token>'.",
        )

    token = parts[1]
    try:
        decoded_token = auth.verify_id_token(token)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=f"Invalid or expired authentication token: {e!s}",
        ) from e

    if not decoded_token.get("email_verified", False):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Email address is not verified. Access forbidden.",
        )

    return decoded_token


def check_user_rate_limit(uid: str) -> None:
    """Enforce per-user sliding window rate limit of 10 requests per minute."""
    now = time.time()
    with _rate_limit_lock:
        timestamps = [
            t for t in _user_request_timestamps[uid]
            if now - t < RATE_LIMIT_WINDOW_SECONDS
        ]
        if len(timestamps) >= USER_RATE_LIMIT_MAX:
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail="Rate limit exceeded. Maximum 10 requests per minute per user.",
            )
        timestamps.append(now)
        _user_request_timestamps[uid] = timestamps


def check_ip_rate_limit(client_ip: str) -> None:
    """Enforce per-IP sliding window rate limit of 10 requests per minute for guests."""
    now = time.time()
    with _rate_limit_lock:
        timestamps = [
            t for t in _ip_request_timestamps[client_ip]
            if now - t < RATE_LIMIT_WINDOW_SECONDS
        ]
        if len(timestamps) >= GUEST_RATE_LIMIT_MAX:
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail="Rate limit exceeded for guest requests. Maximum 10 requests per minute.",
            )
        timestamps.append(now)
        _ip_request_timestamps[client_ip] = timestamps


class HealthResponse(BaseModel):
    """Schema for health status endpoint."""

    status: str = Field("healthy", description="Service health state")
    model_version: str = Field("1.0.0", description="Semantic model version")
    model_name: str = Field("InsightX-MobileNetV2-ONNX", description="Model architecture name")
    runtime: str = Field("onnxruntime", description="Underlying inference engine")
    disclaimer: str = Field(..., description="Mandatory medical research disclaimer")


class PredictResponse(BaseModel):
    """Schema for prediction endpoint."""

    label: str = Field(..., description="Predicted finding label (Normal or Abnormal)")
    prediction_code: int = Field(..., description="0 for Normal, 1 for Abnormal")
    calibrated_probability: float = Field(
        ..., description="Platt-calibrated probability of abnormality [0.0, 1.0]"
    )
    uncalibrated_probability: float = Field(
        ..., description="Raw model sigmoid probability [0.0, 1.0]"
    )
    gradcam_overlay: str = Field(
        ..., description="Base64 data URI of Grad-CAM attention heatmap overlay PNG"
    )
    disclaimer: str = Field(..., description="Mandatory medical research disclaimer")


def _find_file_path(filename: str) -> Path:
    """Resolve file path whether running from project root or api directory."""
    candidates = [
        Path(filename),
        Path("..") / filename,
        Path(__file__).parent.parent / filename,
        Path(__file__).parent / filename,
    ]
    for c in candidates:
        if c.is_file() or c.is_dir():
            return c.resolve()
    return Path(filename)


class ModelArtifacts:
    """Container for in-memory ONNX session and calibration metadata."""

    def __init__(self) -> None:
        self.session: ort.InferenceSession | None = None
        self.input_name: str = "image_input"
        self.output_names: list[str] = []
        self.is_dual_output: bool = False
        self.calib_coef: float = 1.0
        self.calib_intercept: float = 0.0
        self.cam_weights: np.ndarray | None = None

    def load(self) -> None:
        """Load ONNX model, calibration config, and CAM weights once into RAM."""
        dual_path = _find_file_path("models/insightx_gradcam.onnx")
        single_path = _find_file_path("models/insightx.onnx")

        onnx_file = dual_path if dual_path.is_file() else single_path
        if not onnx_file.is_file():
            print(f"[Warning] ONNX model not found at {dual_path} or {single_path}")
            return

        print(f"[API Startup] Loading ONNX model from {onnx_file}...")
        self.session = ort.InferenceSession(str(onnx_file), providers=["CPUExecutionProvider"])
        self.input_name = self.session.get_inputs()[0].name
        self.output_names = [out.name for out in self.session.get_outputs()]
        self.is_dual_output = len(self.output_names) >= 2

        calib_path = _find_file_path("models/calibration.json")
        if calib_path.is_file():
            with open(calib_path, encoding="utf-8") as f:
                calib_data = json.load(f)
            self.calib_coef = float(calib_data.get("coef", 1.0))
            self.calib_intercept = float(calib_data.get("intercept", 0.0))
            print(
                f"[API Startup] Loaded Platt calibration (coef={self.calib_coef:.4f}, "
                f"intercept={self.calib_intercept:.4f})"
            )

        weights_path = _find_file_path("models/gradcam_head_weights.json")
        if weights_path.is_file():
            with open(weights_path, encoding="utf-8") as f:
                w_data = json.load(f)
            self.cam_weights = np.array(w_data.get("weights", []), dtype=np.float32)
            print(f"[API Startup] Loaded {len(self.cam_weights)}-dim CAM weights")


artifacts = ModelArtifacts()


@asynccontextmanager
async def lifespan(_: FastAPI):
    """Lifespan context manager to load model artifacts once at startup."""
    initialize_firebase_admin()
    artifacts.load()
    yield


app = FastAPI(
    title="InsightX - Chest X-Ray Research Demo API",
    description="ONNX-powered AI prediction API with zero-TensorFlow serving.",
    version="1.0.0",
    lifespan=lifespan,
)

# Configure CORS with ALLOWED_ORIGINS environment variable
allowed_origins_env = os.getenv("ALLOWED_ORIGINS", "*")
origins = [o.strip() for o in allowed_origins_env.split(",") if o.strip()]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health", response_model=HealthResponse)
def get_health() -> HealthResponse:
    """Return API health status and model version."""
    return HealthResponse(
        status="healthy",
        model_version="1.0.0",
        model_name="InsightX-MobileNetV2-ONNX",
        runtime="onnxruntime",
        disclaimer=RESEARCH_DISCLAIMER,
    )


@app.get("/samples", response_model=list[str])
def list_samples() -> list[str]:
    """Return a list of available sample image IDs from the samples/ directory."""
    samples_dir = _find_file_path("samples")
    if not samples_dir.is_dir():
        return []
    valid_exts = {".png", ".jpg", ".jpeg"}
    sample_files = [p.name for p in samples_dir.iterdir() if p.suffix.lower() in valid_exts]
    return sorted(sample_files)


@app.get("/samples/{sample_id}")
def get_sample_image(sample_id: str) -> FileResponse:
    """Serve a sample image file from the samples/ directory."""
    samples_dir = _find_file_path("samples")
    target_path = (samples_dir / sample_id).resolve()
    if not str(target_path).startswith(str(samples_dir.resolve())):
        raise HTTPException(status_code=400, detail="Invalid sample path.")
    if not target_path.is_file():
        raise HTTPException(status_code=404, detail="Sample image not found.")
    return FileResponse(target_path, media_type="image/png")


def _preprocess_image_in_memory(image: Image.Image) -> np.ndarray:
    """Preprocess in-memory PIL image for MobileNetV2 without disk I/O."""
    rgb_img = image.convert("RGB")
    resized = rgb_img.resize((224, 224), Image.Resampling.BILINEAR)
    arr = np.asarray(resized, dtype=np.float32)
    preprocessed = (arr / 127.5) - 1.0
    return preprocessed[np.newaxis, ...]  # shape (1, 224, 224, 3)


def _compute_numpy_cam_overlay(
    image: Image.Image,
    conv_features: np.ndarray,
    weights: np.ndarray,
    alpha: float = 0.45,
) -> str:
    """Compute pure NumPy CAM heatmap and return base64 encoded PNG overlay data URI."""
    cam = np.sum(conv_features * weights.reshape(1, 1, -1), axis=-1)
    cam = np.maximum(cam, 0.0)
    max_val = np.max(cam)
    if max_val > 0:
        cam = cam / (max_val + 1e-10)

    w, h = image.size
    cam_uint8 = np.uint8(np.clip(cam * 255.0, 0, 255))
    cam_resized = np.array(
        Image.fromarray(cam_uint8).resize((w, h), Image.Resampling.BILINEAR)
    ) / 255.0

    cmap = plt.get_cmap("jet")
    colored_heatmap = cmap(cam_resized)[..., :3]

    orig_rgb = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
    blended = (1.0 - alpha) * orig_rgb + alpha * colored_heatmap
    overlay_uint8 = np.uint8(np.clip(blended * 255.0, 0, 255))

    buf = io.BytesIO()
    Image.fromarray(overlay_uint8).save(buf, format="PNG")
    b64_str = base64.b64encode(buf.getvalue()).decode("utf-8")
    return f"data:image/png;base64,{b64_str}"


def _run_prediction_pipeline(image: Image.Image) -> PredictResponse:
    """Execute end-to-end inference, Platt calibration, and Grad-CAM overlay."""
    if artifacts.session is None:
        artifacts.load()
        if artifacts.session is None:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Inference model is currently unavailable.",
            )

    img_tensor = _preprocess_image_in_memory(image)
    outputs = artifacts.session.run(None, {artifacts.input_name: img_tensor})

    if artifacts.is_dual_output:
        conv_out = outputs[0] if outputs[0].shape[-1] == 1280 else outputs[1]
        logit_out = outputs[1] if outputs[1].shape[-1] == 1 else outputs[0]
        raw_logit = float(logit_out[0, 0])
        uncalibrated_prob = float(1.0 / (1.0 + np.exp(-raw_logit)))
    else:
        uncalibrated_prob = float(outputs[0][0, 0])
        p_clipped = np.clip(uncalibrated_prob, 1e-6, 1.0 - 1e-6)
        raw_logit = float(np.log(p_clipped / (1.0 - p_clipped)))
        conv_out = None

    calibrated_logit = artifacts.calib_coef * raw_logit + artifacts.calib_intercept
    calibrated_prob = float(1.0 / (1.0 + np.exp(-calibrated_logit)))
    calibrated_prob = float(np.clip(calibrated_prob, 0.0, 1.0))

    pred_code = 1 if calibrated_prob >= 0.5 else 0
    pred_label = "Abnormal" if pred_code == 1 else "Normal"

    if conv_out is not None and artifacts.cam_weights is not None:
        gradcam_b64 = _compute_numpy_cam_overlay(
            image=image,
            conv_features=conv_out[0],
            weights=artifacts.cam_weights,
        )
    else:
        buf = io.BytesIO()
        image.convert("RGB").save(buf, format="PNG")
        b64_str = base64.b64encode(buf.getvalue()).decode("utf-8")
        gradcam_b64 = f"data:image/png;base64,{b64_str}"

    return PredictResponse(
        label=pred_label,
        prediction_code=pred_code,
        calibrated_probability=round(calibrated_prob, 4),
        uncalibrated_probability=round(uncalibrated_prob, 4),
        gradcam_overlay=gradcam_b64,
        disclaimer=RESEARCH_DISCLAIMER,
    )


@app.get("/samples/{sample_id}/predict", response_model=PredictResponse)
def predict_sample_image(
    sample_id: str,
    request: Request,
) -> PredictResponse:
    """Guest endpoint: run inference on a sample image without requiring a token.

    Enforces per-IP sliding window rate limiting (10 req/min).
    """
    client_ip = request.client.host if request.client else "127.0.0.1"
    check_ip_rate_limit(client_ip)

    samples_dir = _find_file_path("samples")
    target_file = None
    candidates = [
        samples_dir / sample_id,
        samples_dir / f"{sample_id}.png",
        samples_dir / f"{sample_id}.jpg",
    ]
    for c in candidates:
        if c.is_file():
            target_file = c
            break

    if not target_file:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Sample '{sample_id}' not found.",
        )

    if not str(target_file.resolve()).startswith(str(samples_dir.resolve())):
        raise HTTPException(status_code=400, detail="Invalid sample path.")

    try:
        image = Image.open(target_file).convert("RGB")
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to load sample image: {e!s}",
        ) from e

    return _run_prediction_pipeline(image)


@app.post("/predict", response_model=PredictResponse)
async def predict_chest_xray(
    file: UploadFile = File(..., description="Chest radiograph image (PNG or JPEG, max 8 MB)"),
    user: dict[str, Any] = Depends(verify_firebase_bearer_token),
) -> PredictResponse:
    """Predict abnormality finding from an in-memory chest radiograph.

    Requires:
    1. Valid Bearer Firebase ID token with verified email.
    2. Adherence to per-user rate limit (10 req/min).
    3. Valid file type (PNG/JPEG), max size (8 MB), and decodability.
    """
    check_user_rate_limit(user["uid"])

    contents = await file.read()
    if len(contents) > MAX_UPLOAD_SIZE_BYTES:
        size_mb = len(contents) / (1024 * 1024)
        raise HTTPException(
            status_code=status.HTTP_413_CONTENT_TOO_LARGE,
            detail=f"File size ({size_mb:.2f} MB) exceeds maximum limit of 8 MB.",
        )

    if len(contents) == 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Uploaded file is empty.",
        )

    content_type = (file.content_type or "").lower()
    if content_type and content_type not in ALLOWED_CONTENT_TYPES:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported file type '{file.content_type}'. Allowed types: PNG, JPEG.",
        )

    try:
        image = Image.open(io.BytesIO(contents))
        image.load()
    except (UnidentifiedImageError, OSError, ValueError) as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Corrupted or non-decodable image file.",
        ) from e

    if image.format not in ALLOWED_IMAGE_FORMATS:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported image format '{image.format}'. Only PNG and JPEG are accepted.",
        )

    return _run_prediction_pipeline(image)
