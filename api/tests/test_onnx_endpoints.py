"""Tests for FastAPI endpoints using the real ONNX model and real sample artifacts."""

import io
import json
from pathlib import Path
from unittest.mock import patch

from fastapi.testclient import TestClient
from PIL import Image

from api.main import _ip_request_timestamps, _user_request_timestamps, app

client = TestClient(app)
REPO_ROOT = Path(__file__).resolve().parent.parent.parent
MANIFEST_PATH = REPO_ROOT / "samples" / "manifest.json"

MOCK_VERIFIED_USER = {
    "uid": "doctor_test_user",
    "email": "doctor@hospital.org",
    "email_verified": True,
}


def _load_manifest() -> list[dict]:
    assert MANIFEST_PATH.is_file(), f"Missing manifest at {MANIFEST_PATH}"
    with open(MANIFEST_PATH, encoding="utf-8") as f:
        return json.load(f)


def test_health_endpoint_real_model():
    """Verify GET /health returns status, version, and valid sha256 file hash."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()

    assert data["status"] == "healthy"
    assert data["model_version"] == "1.0.0"
    assert "file_hash" in data
    assert len(data["file_hash"]) == 64  # valid sha256 hex digest
    assert data["runtime"] == "onnxruntime"
    assert "not a certified medical device" in data["disclaimer"].lower()


def test_samples_catalog_and_images():
    """Verify GET /samples returns manifest items and GET /samples/{id}/image serves images."""
    manifest = _load_manifest()
    response = client.get("/samples")
    assert response.status_code == 200
    catalog = response.json()

    assert len(catalog) == len(manifest) == 8

    for item in catalog:
        assert "id" in item
        assert "ground_truth_label" in item
        assert "thumbnail_url" in item
        assert "calibrated_probability" in item

        # Verify image retrieval endpoint
        img_resp = client.get(item["thumbnail_url"])
        assert img_resp.status_code == 200
        assert img_resp.headers["content-type"] == "image/png"


def test_guest_sample_prediction_matches_manifest():
    """Verify GET /samples/{id}/predict matches calibrated probability in manifest within 0.01."""
    _ip_request_timestamps.clear()
    manifest = _load_manifest()

    for item in manifest:
        sample_id = item["id"]
        expected_prob = item["calibrated_probability"]

        resp = client.get(f"/samples/{sample_id}/predict")
        assert resp.status_code == 200
        data = resp.json()

        actual_prob = data["calibrated_probability"]
        assert 0.0 <= actual_prob <= 1.0
        assert abs(actual_prob - expected_prob) <= 0.01, (
            f"Sample {sample_id} prob {actual_prob} differs from manifest {expected_prob}"
        )

        assert data["threshold_used"] == 0.6522
        assert data["gradcam_overlay"].startswith("data:image/png;base64,")
        assert data["label"] in ["Normal", "Abnormal"]
        assert data["finding_label"] in [
            "Possible abnormal finding",
            "No abnormal finding flagged",
        ]
        assert "diagnosis" not in str(data).lower()
        assert "limitations" in data
        assert "projection_hint" in data


@patch("api.main.auth.verify_id_token")
def test_authenticated_predict_matches_manifest(mock_verify):
    """Verify POST /predict on uploaded sample matches manifest probability within 0.01."""
    _user_request_timestamps.clear()
    mock_verify.return_value = MOCK_VERIFIED_USER
    manifest = _load_manifest()
    sample_entry = manifest[0]
    sample_file = REPO_ROOT / "samples" / sample_entry["id"]

    assert sample_file.is_file(), f"Missing {sample_file}"
    image_bytes = sample_file.read_bytes()

    response = client.post(
        "/predict",
        headers={"Authorization": "Bearer valid_doctor_token"},
        files={"file": (sample_entry["id"], image_bytes, "image/png")},
    )
    assert response.status_code == 200
    data = response.json()

    expected_prob = sample_entry["calibrated_probability"]
    assert abs(data["calibrated_probability"] - expected_prob) <= 0.01
    assert data["threshold_used"] == 0.6522
    assert "diagnosis" not in str(data).lower()


def test_post_predict_missing_token_rejected():
    """Verify POST /predict rejects missing token with 401."""
    img = Image.new("RGB", (224, 224), color=(100, 100, 100))
    buf = io.BytesIO()
    img.save(buf, format="PNG")

    response = client.post(
        "/predict",
        files={"file": ("test.png", buf.getvalue(), "image/png")},
    )
    assert response.status_code == 401


@patch("api.main.auth.verify_id_token")
def test_post_predict_invalid_corrupted_file_rejected(mock_verify):
    """Verify POST /predict rejects corrupted image bytes with 400 Bad Request."""
    mock_verify.return_value = MOCK_VERIFIED_USER
    corrupt_bytes = b"\x89PNG\r\n\x1a\nNotAValidPngStream"

    response = client.post(
        "/predict",
        headers={"Authorization": "Bearer valid_token"},
        files={"file": ("corrupt.png", corrupt_bytes, "image/png")},
    )
    assert response.status_code == 400
    assert "corrupted" in response.json()["detail"].lower()


@patch("api.main.auth.verify_id_token")
def test_post_predict_oversize_file_rejected(mock_verify):
    """Verify POST /predict rejects uploads larger than 8 MB with 413 Content Too Large."""
    mock_verify.return_value = MOCK_VERIFIED_USER
    giant_bytes = b"0" * (8 * 1024 * 1024 + 1024)

    response = client.post(
        "/predict",
        headers={"Authorization": "Bearer valid_token"},
        files={"file": ("giant.png", giant_bytes, "image/png")},
    )
    assert response.status_code == 413
    assert "exceeds" in response.json()["detail"].lower()


@patch("api.main.auth.verify_id_token")
def test_post_predict_overdimension_image_rejected(mock_verify):
    """Verify POST /predict rejects images exceeding 4096 pixels per side with 400."""
    mock_verify.return_value = MOCK_VERIFIED_USER
    big_img = Image.new("L", (4097, 100), color=128)
    buf = io.BytesIO()
    big_img.save(buf, format="PNG")

    response = client.post(
        "/predict",
        headers={"Authorization": "Bearer valid_token"},
        files={"file": ("big.png", buf.getvalue(), "image/png")},
    )
    assert response.status_code == 400
    assert "4096" in response.json()["detail"]
