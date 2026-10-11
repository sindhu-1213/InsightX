"""Unit and integration tests for InsightX FastAPI application and rules compliance.

Validates:
- GET /health endpoint (status, model version, research disclaimer).
- GET /samples endpoint (list of sample IDs).
- Authentication on POST /predict:
  - 401 on missing or malformed Authorization header.
  - 401 on invalid/expired token.
  - 403 on unverified email address.
  - 200 on valid token with verified email.
- Per-user rate limiting on POST /predict (max 10 req/min).
- Guest prediction on GET /samples/{id}/predict without token.
- Per-IP rate limiting on guest prediction endpoint.
- POST /predict validation of file types (PNG/JPEG), corrupt bytes, and size limits (8 MB).
- Strict Rule 1 compliance (no 'diagnosis' allowed in outputs).
"""

import io
from pathlib import Path
from unittest.mock import patch

import pandas as pd
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from api.main import _ip_request_timestamps, _user_request_timestamps, app
from insightx import RESEARCH_DISCLAIMER
from insightx.data import patient_level_split
from insightx.text import clean_medical_text, derive_finding_labels, get_medical_stopwords

client = TestClient(app)

ONNX_MODEL_EXISTS = (
    Path("models/insightx.onnx").is_file()
    or Path("models/insightx_gradcam.onnx").is_file()
)


def _get_test_image_bytes() -> bytes:
    img = Image.new("RGB", (224, 224), color=(128, 128, 128))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()

VALID_TOKEN_PAYLOAD = {
    "uid": "doctor_user_123",
    "email": "doctor@hospital.org",
    "email_verified": True,
}

UNVERIFIED_TOKEN_PAYLOAD = {
    "uid": "unverified_user_456",
    "email": "unverified@hospital.org",
    "email_verified": False,
}


def test_disclaimer_rule_compliance():
    """Verify Rule 1: Research demo disclaimer is defined and forbids 'diagnosis'."""
    assert "not a certified medical device" in RESEARCH_DISCLAIMER.lower()
    assert "diagnosis" not in RESEARCH_DISCLAIMER.lower()
    assert "predictions" in RESEARCH_DISCLAIMER.lower() or "findings" in RESEARCH_DISCLAIMER.lower()


def test_patient_level_split_rule():
    """Verify Rule 3: Always split data by patient uid, never by row."""
    sample_data = {
        "uid": [1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8, 8, 9, 9, 10, 10],
        "projection": [
            "Frontal", "Lateral", "Frontal", "Lateral",
            "Frontal", "Lateral", "Frontal", "Lateral",
            "Frontal", "Lateral", "Frontal", "Lateral",
            "Frontal", "Lateral", "Frontal", "Lateral",
            "Frontal", "Lateral", "Frontal", "Lateral",
        ],
        "filename": [f"img_{i}.png" for i in range(20)],
    }
    df = pd.DataFrame(sample_data)

    train_df, test_df = patient_level_split(df, test_size=0.2, random_state=42)

    train_uids = set(train_df["uid"].unique())
    test_uids = set(test_df["uid"].unique())

    overlap = train_uids.intersection(test_uids)
    assert len(overlap) == 0, f"Patient UIDs leaked across splits: {overlap}"


def test_medical_nlp_preserves_negations():
    """Verify medical stopword filtering preserves clinical negations."""
    stops = get_medical_stopwords()
    assert "no" not in stops
    assert "not" not in stops
    assert "without" not in stops
    assert "negative" not in stops

    raw_text = "No acute cardiopulmonary disease. Lungs are clear without focal consolidation."
    cleaned = clean_medical_text(raw_text)
    assert "no" in cleaned.split()
    assert "without" in cleaned.split()


def test_derive_finding_labels_semantics():
    """Verify label derivation uses radiologist impression and avoids label leakage."""
    normal_case = "Lungs are normal and clear without acute abnormality."
    abnormal_case = "Bilateral pulmonary opacity and pleural effusion consistent with pneumonia."

    labels = derive_finding_labels(pd.Series([normal_case, abnormal_case]))
    assert labels.iloc[0] == 0  # Normal
    assert labels.iloc[1] == 1  # Abnormal


def test_get_health_endpoint():
    """Verify GET /health returns status, model version, and disclaimer."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()

    assert data["status"] == "healthy"
    assert "model_version" in data
    assert data["model_version"] == "1.0.0"
    assert "disclaimer" in data
    assert "not a certified medical device" in data["disclaimer"].lower()
    assert "diagnosis" not in str(data).lower()


def test_get_samples_endpoint():
    """Verify GET /samples lists available sample image IDs."""
    response = client.get("/samples")
    assert response.status_code == 200
    data = response.json()

    assert isinstance(data, list)
    if not data:
        pytest.skip("No real sample images populated in samples/ directory.")
    assert len(data) > 0
    assert any("sample" in s for s in data)


# ---------------------------------------------------------------------------
# Authentication Tests (POST /predict)
# ---------------------------------------------------------------------------


def test_post_predict_missing_token():
    """Verify POST /predict returns 401 when Authorization header is omitted."""
    image_bytes = _get_test_image_bytes()

    response = client.post(
        "/predict",
        files={"file": ("sample.png", image_bytes, "image/png")},
    )
    assert response.status_code == 401
    assert "missing authorization" in response.json()["detail"].lower()


def test_post_predict_malformed_token():
    """Verify POST /predict returns 401 when header format is not 'Bearer <token>'."""
    image_bytes = _get_test_image_bytes()

    response = client.post(
        "/predict",
        headers={"Authorization": "Token not_bearer_prefix"},
        files={"file": ("sample.png", image_bytes, "image/png")},
    )
    assert response.status_code == 401
    assert "invalid authorization header" in response.json()["detail"].lower()


@patch("api.main.auth.verify_id_token")
def test_post_predict_invalid_expired_token(mock_verify):
    """Verify POST /predict returns 401 when verify_id_token raises an exception."""
    mock_verify.side_effect = Exception("Firebase ID token has expired.")
    image_bytes = _get_test_image_bytes()

    response = client.post(
        "/predict",
        headers={"Authorization": "Bearer expired_or_invalid_token"},
        files={"file": ("sample.png", image_bytes, "image/png")},
    )
    assert response.status_code == 401
    assert "invalid or expired" in response.json()["detail"].lower()


@patch("api.main.auth.verify_id_token")
def test_post_predict_unverified_email_forbidden(mock_verify):
    """Verify POST /predict returns 403 when user's email is unverified."""
    mock_verify.return_value = UNVERIFIED_TOKEN_PAYLOAD
    image_bytes = _get_test_image_bytes()

    response = client.post(
        "/predict",
        headers={"Authorization": "Bearer valid_token_unverified_email"},
        files={"file": ("sample.png", image_bytes, "image/png")},
    )
    assert response.status_code == 403
    assert "email address is not verified" in response.json()["detail"].lower()


@patch("api.main.auth.verify_id_token")
def test_post_predict_verified_token_success(mock_verify):
    """Verify POST /predict returns 200 with valid token and verified email."""
    if not ONNX_MODEL_EXISTS:
        pytest.skip("ONNX model file is absent; skipping model inference test.")

    mock_verify.return_value = VALID_TOKEN_PAYLOAD
    _user_request_timestamps["doctor_user_123"].clear()
    image_bytes = _get_test_image_bytes()

    response = client.post(
        "/predict",
        headers={"Authorization": "Bearer valid_verified_token"},
        files={"file": ("sample.png", image_bytes, "image/png")},
    )

    assert response.status_code == 200
    data = response.json()

    assert data["label"] in ["Normal", "Abnormal"]
    assert 0.0 <= data["calibrated_probability"] <= 1.0
    assert 0.0 <= data["uncalibrated_probability"] <= 1.0
    assert data["gradcam_overlay"].startswith("data:image/png;base64,")
    assert "disclaimer" in data
    assert "diagnosis" not in str(data).lower()


@patch("api.main.auth.verify_id_token")
def test_post_predict_user_rate_limit(mock_verify):
    """Verify POST /predict enforces 10 requests per minute per user."""
    if not ONNX_MODEL_EXISTS:
        pytest.skip(
            "ONNX model is absent; skipping rate limit test that depends on inference."
        )

    mock_verify.return_value = {"uid": "rate_limited_user", "email_verified": True}
    _user_request_timestamps["rate_limited_user"].clear()
    image_bytes = _get_test_image_bytes()

    # Make 10 successful requests
    for i in range(10):
        resp = client.post(
            "/predict",
            headers={"Authorization": "Bearer valid_token"},
            files={"file": ("sample.png", image_bytes, "image/png")},
        )
        assert resp.status_code == 200, f"Request {i+1} failed with {resp.status_code}"

    # 11th request must be rejected with 429
    resp_11 = client.post(
        "/predict",
        headers={"Authorization": "Bearer valid_token"},
        files={"file": ("sample.png", image_bytes, "image/png")},
    )
    assert resp_11.status_code == 429
    assert "rate limit exceeded" in resp_11.json()["detail"].lower()


# ---------------------------------------------------------------------------
# Guest Sample Prediction Tests (GET /samples/{id}/predict)
# ---------------------------------------------------------------------------


def test_guest_predict_sample_without_token():
    """Verify guests can test sample predictions without any auth token."""
    sample_path = Path("samples/sample_01_normal.png")
    if not sample_path.is_file() or not ONNX_MODEL_EXISTS:
        pytest.skip("Sample image or ONNX model is absent; skipping guest sample inference test.")

    _ip_request_timestamps["testclient"].clear()
    _ip_request_timestamps["127.0.0.1"].clear()

    response = client.get(f"/samples/{sample_path.name}/predict")
    assert response.status_code == 200
    data = response.json()

    assert data["label"] in ["Normal", "Abnormal"]
    assert 0.0 <= data["calibrated_probability"] <= 1.0
    assert data["gradcam_overlay"].startswith("data:image/png;base64,")
    assert "disclaimer" in data
    assert "diagnosis" not in str(data).lower()


def test_guest_predict_sample_not_found():
    """Verify 404 error when requesting a non-existent sample ID."""
    response = client.get("/samples/non_existent_image_xyz.png/predict")
    assert response.status_code == 404
    assert "not found" in response.json()["detail"].lower()


def test_guest_predict_ip_rate_limit():
    """Verify guests are rate limited per client IP (10 req/min)."""
    sample_path = Path("samples/sample_01_normal.png")
    if not sample_path.is_file() or not ONNX_MODEL_EXISTS:
        pytest.skip(
            "Sample image or ONNX model is absent; skipping guest rate limit inference test."
        )

    client_ip = "testclient"
    _ip_request_timestamps[client_ip].clear()

    # Perform 10 guest predictions
    for i in range(10):
        resp = client.get(f"/samples/{sample_path.name}/predict")
        assert resp.status_code == 200, f"Guest request {i+1} failed with {resp.status_code}"

    # 11th request triggers 429
    resp_11 = client.get(f"/samples/{sample_path.name}/predict")
    assert resp_11.status_code == 429
    assert "rate limit exceeded" in resp_11.json()["detail"].lower()


# ---------------------------------------------------------------------------
# File Validation Tests (POST /predict)
# ---------------------------------------------------------------------------


@patch("api.main.auth.verify_id_token")
def test_post_predict_invalid_mime_type(mock_verify):
    """Verify POST /predict rejects non-image files with 400 Bad Request."""
    mock_verify.return_value = VALID_TOKEN_PAYLOAD
    _user_request_timestamps["doctor_user_123"].clear()

    fake_text = b"This is a text file, not a chest X-ray radiograph."
    response = client.post(
        "/predict",
        headers={"Authorization": "Bearer valid_token"},
        files={"file": ("report.txt", fake_text, "text/plain")},
    )
    assert response.status_code == 400
    detail = response.json()["detail"].lower()
    assert "unsupported" in detail or "allowed" in detail


@patch("api.main.auth.verify_id_token")
def test_post_predict_corrupted_image(mock_verify):
    """Verify POST /predict rejects corrupted image bytes with 400 Bad Request."""
    mock_verify.return_value = VALID_TOKEN_PAYLOAD
    _user_request_timestamps["doctor_user_123"].clear()

    corrupted_bytes = b"\x89PNG\r\n\x1a\n\x00\x00\x00corrupted_truncated_bytes"
    response = client.post(
        "/predict",
        headers={"Authorization": "Bearer valid_token"},
        files={"file": ("corrupt.png", corrupted_bytes, "image/png")},
    )
    assert response.status_code == 400
    detail = response.json()["detail"].lower()
    assert "corrupted" in detail or "invalid" in detail


@patch("api.main.auth.verify_id_token")
def test_post_predict_oversized_image(mock_verify):
    """Verify POST /predict rejects files exceeding 8 MB with 413 Payload Too Large."""
    mock_verify.return_value = VALID_TOKEN_PAYLOAD
    _user_request_timestamps["doctor_user_123"].clear()

    oversized_bytes = b"0" * (8 * 1024 * 1024 + 1024)
    response = client.post(
        "/predict",
        headers={"Authorization": "Bearer valid_token"},
        files={"file": ("giant.png", oversized_bytes, "image/png")},
    )
    assert response.status_code == 413
    detail = response.json()["detail"].lower()
    assert "exceeds" in detail or "8 mb" in detail
