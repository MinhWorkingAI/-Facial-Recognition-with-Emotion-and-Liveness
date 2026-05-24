"""
conftest.py
===========
Shared pytest fixtures for both unit and integration tests.
Lives in API_endpoint_test/ — shared across:
    API_endpoint_unit_test/
    API_endpoint_integration_test/

Test image defaults:
    Unit tests      → test_assets/Tom_Cruise.jpg
    Integration     → test_assets/ (specific images per test)
"""

from __future__ import annotations

import io
from pathlib import Path

import pytest
import requests
from PIL import Image


# ------------------------------------------------------------------
# Test assets directory
# Resolved relative to this file's location so it works regardless
# of where pytest is invoked from.
# ------------------------------------------------------------------

TEST_ASSETS_DIR = Path(__file__).resolve().parent / "test_assets"
UNIT_TEST_IMAGE = TEST_ASSETS_DIR / "unit_test_img.jpg"


# ------------------------------------------------------------------
# CLI option — base URL only
# ------------------------------------------------------------------

def pytest_addoption(parser):
    parser.addoption(
        "--base-url",
        action="store",
        default="http://127.0.0.1:8000",
        help="Base URL of the running FaceGuard API server",
    )


# ------------------------------------------------------------------
# Fixtures
# ------------------------------------------------------------------

@pytest.fixture(scope="session")
def base_url(request) -> str:
    return request.config.getoption("--base-url").rstrip("/")


@pytest.fixture(scope="session")
def session() -> requests.Session:
    s = requests.Session()
    yield s
    s.close()


@pytest.fixture(scope="session")
def sample_image() -> bytes:
    """
    Programmatically generated 100x100 JPEG.
    Used for structural tests that don't need a real face.
    """
    img = Image.new("RGB", (100, 100), color=(200, 150, 100))
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    buf.seek(0)
    return buf.read()


@pytest.fixture(scope="session")
def blank_image() -> bytes:
    """
    Plain white 100x100 JPEG.
    Used for edge case tests — endpoints should not crash on this.
    """
    img = Image.new("RGB", (100, 100), color=(255, 255, 255))
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    buf.seek(0)
    return buf.read()


@pytest.fixture(scope="session")
def bad_file() -> bytes:
    """Plain text bytes — used to test 400 responses."""
    return b"this is not an image file"


@pytest.fixture(scope="session")
def face_image_bytes() -> bytes:
    """
    Real face image for unit tests.
    Hardcoded to test_assets/unit_test_img.jpg.
    Skips gracefully if file is missing.
    """
    if not UNIT_TEST_IMAGE.is_file():
        pytest.skip(
            f"Unit test image not found at {UNIT_TEST_IMAGE} — "
            f"add {UNIT_TEST_IMAGE.name} to test_assets/"
        )
    return UNIT_TEST_IMAGE.read_bytes()


@pytest.fixture(scope="session")
def server_available(base_url, session) -> bool:
    try:
        r = session.get(f"{base_url}/api/health", timeout=5)
        return r.status_code == 200
    except requests.exceptions.ConnectionError:
        return False


@pytest.fixture(autouse=True)
def skip_if_server_down(server_available):
    if not server_available:
        pytest.skip("Server not reachable — start uvicorn before running tests")
