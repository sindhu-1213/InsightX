"""Tests ensuring completeness of documentation and synchronization of published metrics."""

import json
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent.parent


@pytest.mark.parametrize(
    "relative_path",
    [
        "MODEL_CARD.md",
        "README.md",
        "web/src/data/metrics.json",
    ],
)
def test_no_pending_or_todo_placeholders(relative_path: str):
    """Assert that documentation and metrics have no PENDING or TODO placeholders."""
    file_path = REPO_ROOT / relative_path
    assert file_path.is_file(), f"Expected file does not exist: {file_path}"

    content = file_path.read_text(encoding="utf-8")
    assert "PENDING" not in content, f"Found 'PENDING' placeholder in {relative_path}"
    assert "TODO" not in content, f"Found 'TODO' placeholder in {relative_path}"


def test_web_metrics_match_reports_metrics():
    """Assert web/src/data/metrics.json matches reports/metrics.json."""
    reports_metrics_path = REPO_ROOT / "reports" / "metrics.json"
    web_metrics_path = REPO_ROOT / "web" / "src" / "data" / "metrics.json"

    assert reports_metrics_path.is_file(), f"Missing {reports_metrics_path}"
    assert web_metrics_path.is_file(), f"Missing {web_metrics_path}"

    with open(reports_metrics_path, encoding="utf-8") as f:
        reports_data = json.load(f)

    with open(web_metrics_path, encoding="utf-8") as f:
        web_data = json.load(f)

    # Validate that all metric values and breakdowns match
    assert (
        web_data == reports_data
    ), "Mismatch between web/src/data/metrics.json and reports/metrics.json"
