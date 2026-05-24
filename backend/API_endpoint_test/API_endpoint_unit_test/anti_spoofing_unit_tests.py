"""
anti_spoofing_unit_tests.py
===========================
Unit tests for POST /api/anti-spoofing/anti-spoof

Expected response shape:
{
    "anti_spoofing": {
        "label":      "real" | "spoof",
        "confidence": float
    }
}
"""

from __future__ import annotations

import pytest

ENDPOINT = "/api/anti-spoofing/anti-spoof"

VALID_LABELS = {"real", "spoof"}


def post_image(session, base_url, image_bytes, filename="test.jpg", content_type="image/jpeg"):
    return session.post(
        f"{base_url}{ENDPOINT}",
        files={"file": (filename, image_bytes, content_type)},
        timeout=30,
    )


# ------------------------------------------------------------------
# Status codes
# ------------------------------------------------------------------

def test_returns_200_on_valid_image(base_url, session, sample_image):
    r = post_image(session, base_url, sample_image)
    assert r.status_code == 200


def test_returns_400_on_non_image_file(base_url, session, bad_file):
    r = post_image(session, base_url, bad_file,
                   filename="bad.txt", content_type="text/plain")
    assert r.status_code == 400


def test_returns_422_on_no_file(base_url, session):
    r = session.post(f"{base_url}{ENDPOINT}", timeout=30)
    assert r.status_code == 422


# ------------------------------------------------------------------
# Response structure
# ------------------------------------------------------------------

def test_response_is_json(base_url, session, sample_image):
    r = post_image(session, base_url, sample_image)
    assert r.headers["content-type"].startswith("application/json")


def test_response_has_anti_spoofing_key(base_url, session, sample_image):
    r = post_image(session, base_url, sample_image)
    assert "anti_spoofing" in r.json()


def test_anti_spoofing_has_label(base_url, session, sample_image):
    data = post_image(session, base_url, sample_image).json()
    assert "label" in data["anti_spoofing"]


def test_anti_spoofing_has_confidence(base_url, session, sample_image):
    data = post_image(session, base_url, sample_image).json()
    assert "confidence" in data["anti_spoofing"]


def test_anti_spoofing_label_is_string(base_url, session, sample_image):
    data = post_image(session, base_url, sample_image).json()
    assert isinstance(data["anti_spoofing"]["label"], str)


def test_anti_spoofing_label_is_not_empty(base_url, session, sample_image):
    data = post_image(session, base_url, sample_image).json()
    assert len(data["anti_spoofing"]["label"]) > 0


def test_anti_spoofing_label_is_valid(base_url, session, sample_image):
    data = post_image(session, base_url, sample_image).json()
    assert data["anti_spoofing"]["label"] in VALID_LABELS


def test_anti_spoofing_confidence_is_float(base_url, session, sample_image):
    data = post_image(session, base_url, sample_image).json()
    assert isinstance(data["anti_spoofing"]["confidence"], float)


def test_anti_spoofing_confidence_in_range(base_url, session, sample_image):
    data = post_image(session, base_url, sample_image).json()
    assert 0.0 <= data["anti_spoofing"]["confidence"] <= 1.0


# ------------------------------------------------------------------
# Edge cases
# ------------------------------------------------------------------

def test_blank_image_returns_200(base_url, session, blank_image):
    r = post_image(session, base_url, blank_image)
    assert r.status_code == 200


def test_blank_image_returns_valid_structure(base_url, session, blank_image):
    data = post_image(session, base_url, blank_image).json()
    assert "anti_spoofing" in data
    assert "label" in data["anti_spoofing"]
    assert "confidence" in data["anti_spoofing"]
