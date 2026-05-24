"""
detection_unit_tests.py
=======================
Unit tests for POST /api/detection/detect

Expected response shape:
{
    "image_width":  int,
    "image_height": int,
    "faces": [
        {
            "bbox": {"x": float, "y": float, "w": float, "h": float},
            "detection_confidence": float,
            "crop_width":  int,
            "crop_height": int
        }
    ]
}
"""

from __future__ import annotations

import pytest

ENDPOINT = "/api/detection/detect"


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
# Top-level response structure
# ------------------------------------------------------------------

def test_response_is_json(base_url, session, sample_image):
    r = post_image(session, base_url, sample_image)
    assert r.headers["content-type"].startswith("application/json")


def test_response_has_image_width(base_url, session, sample_image):
    r = post_image(session, base_url, sample_image)
    assert "image_width" in r.json()


def test_response_has_image_height(base_url, session, sample_image):
    r = post_image(session, base_url, sample_image)
    assert "image_height" in r.json()


def test_response_has_faces(base_url, session, sample_image):
    r = post_image(session, base_url, sample_image)
    assert "faces" in r.json()


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


def test_blank_image_returns_empty_faces_list(base_url, session, blank_image):
    data = post_image(session, base_url, blank_image).json()
    assert data["faces"] == []


# ------------------------------------------------------------------
# Per-face structure — requires real face image
# ------------------------------------------------------------------

def test_face_has_bbox(base_url, session, face_image_bytes):
    data = post_image(session, base_url, face_image_bytes).json()
    assert len(data["faces"]) >= 1
    for face in data["faces"]:
        assert "bbox" in face


def test_face_has_detection_confidence(base_url, session, face_image_bytes):
    data = post_image(session, base_url, face_image_bytes).json()
    assert len(data["faces"]) >= 1
    for face in data["faces"]:
        assert "detection_confidence" in face


def test_face_has_crop_width(base_url, session, face_image_bytes):
    data = post_image(session, base_url, face_image_bytes).json()
    assert len(data["faces"]) >= 1
    for face in data["faces"]:
        assert "crop_width" in face


def test_face_has_crop_height(base_url, session, face_image_bytes):
    data = post_image(session, base_url, face_image_bytes).json()
    assert len(data["faces"]) >= 1
    for face in data["faces"]:
        assert "crop_height" in face


def test_bbox_has_x(base_url, session, face_image_bytes):
    data = post_image(session, base_url, face_image_bytes).json()
    for face in data["faces"]:
        assert "x" in face["bbox"]


def test_bbox_has_y(base_url, session, face_image_bytes):
    data = post_image(session, base_url, face_image_bytes).json()
    for face in data["faces"]:
        assert "y" in face["bbox"]


def test_bbox_has_w(base_url, session, face_image_bytes):
    data = post_image(session, base_url, face_image_bytes).json()
    for face in data["faces"]:
        assert "w" in face["bbox"]


def test_bbox_has_h(base_url, session, face_image_bytes):
    data = post_image(session, base_url, face_image_bytes).json()
    for face in data["faces"]:
        assert "h" in face["bbox"]


def test_bbox_x_is_normalized(base_url, session, face_image_bytes):
    data = post_image(session, base_url, face_image_bytes).json()
    for face in data["faces"]:
        assert 0.0 <= face["bbox"]["x"] <= 1.0


def test_bbox_y_is_normalized(base_url, session, face_image_bytes):
    data = post_image(session, base_url, face_image_bytes).json()
    for face in data["faces"]:
        assert 0.0 <= face["bbox"]["y"] <= 1.0


def test_bbox_w_is_normalized(base_url, session, face_image_bytes):
    data = post_image(session, base_url, face_image_bytes).json()
    for face in data["faces"]:
        assert 0.0 <= face["bbox"]["w"] <= 1.0


def test_bbox_h_is_normalized(base_url, session, face_image_bytes):
    data = post_image(session, base_url, face_image_bytes).json()
    for face in data["faces"]:
        assert 0.0 <= face["bbox"]["h"] <= 1.0


def test_bbox_x_plus_w_does_not_exceed_1(base_url, session, face_image_bytes):
    data = post_image(session, base_url, face_image_bytes).json()
    for face in data["faces"]:
        assert face["bbox"]["x"] + face["bbox"]["w"] <= 1.0 + 1e-5


def test_bbox_y_plus_h_does_not_exceed_1(base_url, session, face_image_bytes):
    data = post_image(session, base_url, face_image_bytes).json()
    for face in data["faces"]:
        assert face["bbox"]["y"] + face["bbox"]["h"] <= 1.0 + 1e-5


def test_detection_confidence_is_float(base_url, session, face_image_bytes):
    data = post_image(session, base_url, face_image_bytes).json()
    for face in data["faces"]:
        assert isinstance(face["detection_confidence"], float)


def test_detection_confidence_in_range(base_url, session, face_image_bytes):
    data = post_image(session, base_url, face_image_bytes).json()
    for face in data["faces"]:
        assert 0.0 <= face["detection_confidence"] <= 1.0


def test_crop_width_is_positive_int(base_url, session, face_image_bytes):
    data = post_image(session, base_url, face_image_bytes).json()
    for face in data["faces"]:
        assert isinstance(face["crop_width"], int)
        assert face["crop_width"] > 0


def test_crop_height_is_positive_int(base_url, session, face_image_bytes):
    data = post_image(session, base_url, face_image_bytes).json()
    for face in data["faces"]:
        assert isinstance(face["crop_height"], int)
        assert face["crop_height"] > 0
