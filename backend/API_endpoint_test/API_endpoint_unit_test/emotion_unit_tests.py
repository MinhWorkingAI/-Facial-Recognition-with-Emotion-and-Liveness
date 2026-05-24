"""
emotion_unit_tests.py
=====================
Unit tests for POST /api/emotion/

Expected response shape:
{
    "emotion": {
        "label":      str,
        "confidence": float
    }
}
"""

from __future__ import annotations

import pytest

ENDPOINT = "/api/emotion/"

VALID_LABELS = {
    "anger", "contempt", "disgust", "fear",
    "happy", "neutral", "sad", "surprise",
}


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


def test_response_has_emotion_key(base_url, session, sample_image):
    r = post_image(session, base_url, sample_image)
    assert "emotion" in r.json()


def test_emotion_has_label(base_url, session, sample_image):
    data = post_image(session, base_url, sample_image).json()
    assert "label" in data["emotion"]


def test_emotion_has_confidence(base_url, session, sample_image):
    data = post_image(session, base_url, sample_image).json()
    assert "confidence" in data["emotion"]


def test_emotion_label_is_string(base_url, session, sample_image):
    data = post_image(session, base_url, sample_image).json()
    assert isinstance(data["emotion"]["label"], str)


def test_emotion_label_is_not_empty(base_url, session, sample_image):
    data = post_image(session, base_url, sample_image).json()
    assert len(data["emotion"]["label"]) > 0


def test_emotion_label_is_valid(base_url, session, sample_image):
    data = post_image(session, base_url, sample_image).json()
    assert data["emotion"]["label"] in VALID_LABELS


def test_emotion_confidence_is_float(base_url, session, sample_image):
    data = post_image(session, base_url, sample_image).json()
    assert isinstance(data["emotion"]["confidence"], float)


def test_emotion_confidence_in_range(base_url, session, sample_image):
    data = post_image(session, base_url, sample_image).json()
    assert 0.0 <= data["emotion"]["confidence"] <= 1.0


# ------------------------------------------------------------------
# Edge cases
# ------------------------------------------------------------------

def test_blank_image_returns_200(base_url, session, blank_image):
    r = post_image(session, base_url, blank_image)
    assert r.status_code == 200


def test_blank_image_returns_valid_structure(base_url, session, blank_image):
    data = post_image(session, base_url, blank_image).json()
    assert "emotion" in data
    assert "label" in data["emotion"]
    assert "confidence" in data["emotion"]
