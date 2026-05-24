"""
verification_unit_tests.py
==========================
Unit tests for:
    POST /api/verification/register
    POST /api/verification/verify

Register response shape:
{
    "person_id": str,
    "status":    "registered" | "updated",
    "message":   str
}

Verify response shape:
{
    "recognition": {
        "label":      str,
        "confidence": float,
        "matched":    bool
    }
}

Note: register and verify use separate service instances in the current
codebase — a registered face via /register will NOT be seen by /verify.
This is a known issue tracked separately.
"""

from __future__ import annotations

import pytest

REGISTER_ENDPOINT = "/api/verification/register"
VERIFY_ENDPOINT   = "/api/verification/verify"

PERSON_ID = "unit_test_employee"


def post_register(session, base_url, image_bytes, person_id=PERSON_ID,
                  filename="test.jpg", content_type="image/jpeg"):
    return session.post(
        f"{base_url}{REGISTER_ENDPOINT}",
        data={"person_id": person_id},
        files={"file": (filename, image_bytes, content_type)},
        timeout=30,
    )


def post_verify(session, base_url, image_bytes,
                filename="test.jpg", content_type="image/jpeg"):
    return session.post(
        f"{base_url}{VERIFY_ENDPOINT}",
        files={"file": (filename, image_bytes, content_type)},
        timeout=30,
    )


# ------------------------------------------------------------------
# Register — status codes
# ------------------------------------------------------------------

def test_register_returns_200_on_valid_image(base_url, session, face_image_bytes):
    r = post_register(session, base_url, face_image_bytes)
    assert r.status_code == 200


def test_register_returns_400_on_non_image_file(base_url, session, bad_file):
    r = post_register(session, base_url, bad_file,
                      filename="bad.txt", content_type="text/plain")
    assert r.status_code == 400


def test_register_returns_422_on_missing_person_id(base_url, session, face_image_bytes):
    r = session.post(
        f"{base_url}{REGISTER_ENDPOINT}",
        files={"file": ("test.jpg", face_image_bytes, "image/jpeg")},
        timeout=30,
    )
    assert r.status_code == 422


def test_register_returns_422_on_no_file(base_url, session):
    r = session.post(
        f"{base_url}{REGISTER_ENDPOINT}",
        data={"person_id": PERSON_ID},
        timeout=30,
    )
    assert r.status_code == 422


# ------------------------------------------------------------------
# Register — response structure
# ------------------------------------------------------------------

def test_register_response_is_json(base_url, session, face_image_bytes):
    r = post_register(session, base_url, face_image_bytes)
    assert r.headers["content-type"].startswith("application/json")


def test_register_has_person_id(base_url, session, face_image_bytes):
    data = post_register(session, base_url, face_image_bytes).json()
    assert "person_id" in data


def test_register_has_status(base_url, session, face_image_bytes):
    data = post_register(session, base_url, face_image_bytes).json()
    assert "status" in data


def test_register_has_message(base_url, session, face_image_bytes):
    data = post_register(session, base_url, face_image_bytes).json()
    assert "message" in data


def test_register_person_id_matches_input(base_url, session, face_image_bytes):
    data = post_register(session, base_url, face_image_bytes, person_id=PERSON_ID).json()
    assert data["person_id"] == PERSON_ID


def test_register_status_is_valid(base_url, session, face_image_bytes):
    data = post_register(session, base_url, face_image_bytes).json()
    assert data["status"] in ("registered", "updated")


def test_register_twice_returns_updated(base_url, session, face_image_bytes):
    post_register(session, base_url, face_image_bytes, person_id=PERSON_ID)
    data = post_register(session, base_url, face_image_bytes, person_id=PERSON_ID).json()
    assert data["status"] == "updated"


def test_register_message_is_string(base_url, session, face_image_bytes):
    data = post_register(session, base_url, face_image_bytes).json()
    assert isinstance(data["message"], str)
    assert len(data["message"]) > 0


# ------------------------------------------------------------------
# Verify — status codes
# ------------------------------------------------------------------

def test_verify_returns_200_on_valid_image(base_url, session, face_image_bytes):
    r = post_verify(session, base_url, face_image_bytes)
    assert r.status_code == 200


def test_verify_returns_400_on_non_image_file(base_url, session, bad_file):
    r = post_verify(session, base_url, bad_file,
                    filename="bad.txt", content_type="text/plain")
    assert r.status_code == 400


def test_verify_returns_422_on_no_file(base_url, session):
    r = session.post(f"{base_url}{VERIFY_ENDPOINT}", timeout=30)
    assert r.status_code == 422


# ------------------------------------------------------------------
# Verify — response structure
# ------------------------------------------------------------------

def test_verify_response_is_json(base_url, session, face_image_bytes):
    r = post_verify(session, base_url, face_image_bytes)
    assert r.headers["content-type"].startswith("application/json")


def test_verify_has_recognition_key(base_url, session, face_image_bytes):
    data = post_verify(session, base_url, face_image_bytes).json()
    assert "recognition" in data


def test_verify_recognition_has_label(base_url, session, face_image_bytes):
    data = post_verify(session, base_url, face_image_bytes).json()
    assert "label" in data["recognition"]


def test_verify_recognition_has_confidence(base_url, session, face_image_bytes):
    data = post_verify(session, base_url, face_image_bytes).json()
    assert "confidence" in data["recognition"]


def test_verify_recognition_has_matched(base_url, session, face_image_bytes):
    data = post_verify(session, base_url, face_image_bytes).json()
    assert "matched" in data["recognition"]


def test_verify_label_is_string(base_url, session, face_image_bytes):
    data = post_verify(session, base_url, face_image_bytes).json()
    assert isinstance(data["recognition"]["label"], str)


def test_verify_label_is_not_empty(base_url, session, face_image_bytes):
    data = post_verify(session, base_url, face_image_bytes).json()
    assert len(data["recognition"]["label"]) > 0


def test_verify_confidence_is_float(base_url, session, face_image_bytes):
    data = post_verify(session, base_url, face_image_bytes).json()
    assert isinstance(data["recognition"]["confidence"], float)


def test_verify_confidence_in_range(base_url, session, face_image_bytes):
    data = post_verify(session, base_url, face_image_bytes).json()
    assert 0.0 <= data["recognition"]["confidence"] <= 1.0


def test_verify_matched_is_bool(base_url, session, face_image_bytes):
    data = post_verify(session, base_url, face_image_bytes).json()
    assert isinstance(data["recognition"]["matched"], bool)


def test_verify_matched_is_bool_after_registration(base_url, session, face_image_bytes):
    """
    After registration tests have run, verify returns a valid bool for matched.
    We only check the type — not the value — since state depends on test order.
    """
    data = post_verify(session, base_url, face_image_bytes).json()
    assert isinstance(data["recognition"]["matched"], bool)
