"""
pipeline_unit_tests.py
======================
Unit tests for POST /api/pipeline/frame

Expected response shape:
{
    "image_width":  int,
    "image_height": int,
    "faces": [
        {
            "face": {
                "bbox": {"x": float, "y": float, "w": float, "h": float},
                "detection_confidence": float,
                "crop_width":  int,
                "crop_height": int
            },
            "emotion": {
                "label":      str,
                "confidence": float
            },
            "anti_spoofing": {
                "label":      "real" | "spoof",
                "confidence": float
            },
            "recognition": {
                "label":      str,
                "confidence": float,
                "matched":    bool
            }
        }
    ]
}
"""

from __future__ import annotations

import pytest

ENDPOINT = "/api/pipeline/frame"

VALID_EMOTION_LABELS      = {"anger", "contempt", "disgust", "fear",
                              "happy", "neutral", "sad", "surprise"}
VALID_ANTI_SPOOFING_LABELS = {"real", "spoof"}


def post_image(session, base_url, image_bytes, filename="test.jpg", content_type="image/jpeg"):
    return session.post(
        f"{base_url}{ENDPOINT}",
        files={"file": (filename, image_bytes, content_type)},
        timeout=60,
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
    r = session.post(f"{base_url}{ENDPOINT}", timeout=60)
    assert r.status_code == 422


# ------------------------------------------------------------------
# Top-level response structure
# ------------------------------------------------------------------

def test_response_is_json(base_url, session, sample_image):
    r = post_image(session, base_url, sample_image)
    assert r.headers["content-type"].startswith("application/json")


def test_response_has_image_width(base_url, session, sample_image):
    data = post_image(session, base_url, sample_image).json()
    assert "image_width" in data


def test_response_has_image_height(base_url, session, sample_image):
    data = post_image(session, base_url, sample_image).json()
    assert "image_height" in data


def test_response_has_faces(base_url, session, sample_image):
    data = post_image(session, base_url, sample_image).json()
    assert "faces" in data


def test_image_width_is_positive_int(base_url, session, sample_image):
    data = post_image(session, base_url, sample_image).json()
    assert isinstance(data["image_width"], int)
    assert data["image_width"] > 0


def test_image_height_is_positive_int(base_url, session, sample_image):
    data = post_image(session, base_url, sample_image).json()
    assert isinstance(data["image_height"], int)
    assert data["image_height"] > 0


def test_faces_is_a_list(base_url, session, sample_image):
    data = post_image(session, base_url, sample_image).json()
    assert isinstance(data["faces"], list)


# ------------------------------------------------------------------
# Edge cases
# ------------------------------------------------------------------

def test_blank_image_returns_200(base_url, session, blank_image):
    r = post_image(session, base_url, blank_image)
    assert r.status_code == 200


def test_blank_image_returns_empty_faces(base_url, session, blank_image):
    data = post_image(session, base_url, blank_image).json()
    assert data["faces"] == []


# ------------------------------------------------------------------
# Per-face structure — requires real face image
# ------------------------------------------------------------------

def test_face_has_face_key(base_url, session, face_image_bytes):
    data = post_image(session, base_url, face_image_bytes).json()
    assert len(data["faces"]) >= 1
    for face in data["faces"]:
        assert "face" in face


def test_face_has_emotion_key(base_url, session, face_image_bytes):
    data = post_image(session, base_url, face_image_bytes).json()
    assert len(data["faces"]) >= 1
    for face in data["faces"]:
        assert "emotion" in face


def test_face_has_anti_spoofing_key(base_url, session, face_image_bytes):
    data = post_image(session, base_url, face_image_bytes).json()
    assert len(data["faces"]) >= 1
    for face in data["faces"]:
        assert "anti_spoofing" in face


def test_face_has_recognition_key(base_url, session, face_image_bytes):
    data = post_image(session, base_url, face_image_bytes).json()
    assert len(data["faces"]) >= 1
    for face in data["faces"]:
        assert "recognition" in face


def test_face_bbox_has_x(base_url, session, face_image_bytes):
    data = post_image(session, base_url, face_image_bytes).json()
    for face in data["faces"]:
        assert "x" in face["face"]["bbox"]


def test_face_bbox_has_y(base_url, session, face_image_bytes):
    data = post_image(session, base_url, face_image_bytes).json()
    for face in data["faces"]:
        assert "y" in face["face"]["bbox"]


def test_face_bbox_has_w(base_url, session, face_image_bytes):
    data = post_image(session, base_url, face_image_bytes).json()
    for face in data["faces"]:
        assert "w" in face["face"]["bbox"]


def test_face_bbox_has_h(base_url, session, face_image_bytes):
    data = post_image(session, base_url, face_image_bytes).json()
    for face in data["faces"]:
        assert "h" in face["face"]["bbox"]


def test_face_bbox_values_normalized(base_url, session, face_image_bytes):
    data = post_image(session, base_url, face_image_bytes).json()
    for face in data["faces"]:
        bbox = face["face"]["bbox"]
        assert 0.0 <= bbox["x"] <= 1.0
        assert 0.0 <= bbox["y"] <= 1.0
        assert 0.0 <= bbox["w"] <= 1.0
        assert 0.0 <= bbox["h"] <= 1.0


def test_detection_confidence_is_float(base_url, session, face_image_bytes):
    data = post_image(session, base_url, face_image_bytes).json()
    for face in data["faces"]:
        assert isinstance(face["face"]["detection_confidence"], float)


def test_detection_confidence_in_range(base_url, session, face_image_bytes):
    data = post_image(session, base_url, face_image_bytes).json()
    for face in data["faces"]:
        assert 0.0 <= face["face"]["detection_confidence"] <= 1.0


def test_emotion_label_is_valid(base_url, session, face_image_bytes):
    data = post_image(session, base_url, face_image_bytes).json()
    for face in data["faces"]:
        assert face["emotion"]["label"] in VALID_EMOTION_LABELS


def test_emotion_confidence_in_range(base_url, session, face_image_bytes):
    data = post_image(session, base_url, face_image_bytes).json()
    for face in data["faces"]:
        conf = face["emotion"]["confidence"]
        assert isinstance(conf, float)
        assert 0.0 <= conf <= 1.0


def test_anti_spoofing_label_is_valid(base_url, session, face_image_bytes):
    data = post_image(session, base_url, face_image_bytes).json()
    for face in data["faces"]:
        assert face["anti_spoofing"]["label"] in VALID_ANTI_SPOOFING_LABELS


def test_anti_spoofing_confidence_in_range(base_url, session, face_image_bytes):
    data = post_image(session, base_url, face_image_bytes).json()
    for face in data["faces"]:
        conf = face["anti_spoofing"]["confidence"]
        assert isinstance(conf, float)
        assert 0.0 <= conf <= 1.0


def test_recognition_label_is_string(base_url, session, face_image_bytes):
    data = post_image(session, base_url, face_image_bytes).json()
    for face in data["faces"]:
        assert isinstance(face["recognition"]["label"], str)


# def test_recognition_confidence_in_range(base_url, session, face_image_bytes):
#     data = post_image(session, base_url, face_image_bytes).json()
#     for face in data["faces"]:
#         conf = face["recognition"]["confidence"]
#         assert isinstance(conf, float)
#         assert 0.0 <= conf <= 1.0


def test_recognition_matched_is_bool(base_url, session, face_image_bytes):
    data = post_image(session, base_url, face_image_bytes).json()
    for face in data["faces"]:
        assert isinstance(face["recognition"]["matched"], bool)
