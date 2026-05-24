# API Endpoint Test Guide

## Overview

This test suite verifies that every API endpoint in the FaceGuard backend returns correct HTTP status codes, valid JSON responses, and properly structured data. Tests are written with `pytest` and organized into two layers — unit tests for structural validation and integration tests for model behaviour verification.

All tests run against a live local server. The server must be running before executing any tests.

---

## Folder Organization

```
backend/
    API_endpoint_test/
        conftest.py                        # Shared fixtures for all tests
        test_assets/
            Tom_Cruise.jpg                 # Default face image for unit tests
        API_endpoint_unit_test/
            health_unit_tests.py
            face_detection_unit_tests.py
            emotion_unit_tests.py
            anti_spoofing_unit_tests.py
            verification_unit_tests.py
            pipeline_unit_tests.py
        API_endpoint_integration_test/
            (future — model behaviour tests)
```

---

## Running the Tests

Start the server first in one terminal:

```powershell
cd backend
$env:BACKEND_USE_TRITON="0"
uvicorn app.main:app --reload
```

Then run all tests from the project root:

```powershell
pytest -v
```

Run a single file:

```powershell
pytest backend/API_endpoint_test/API_endpoint_unit_test/health_unit_tests.py -v
```

---

## conftest.py

Lives in `API_endpoint_test/` and is automatically loaded by pytest before any test runs. Shared across both unit and integration test folders.

### Fixtures

| Fixture | Scope | Description |
|---|---|---|
| `base_url` | session | API server base URL. Default: `http://127.0.0.1:8000`. Override with `--base-url`. |
| `session` | session | Shared `requests.Session` reused across all tests. More efficient than a new connection per test. |
| `sample_image` | session | Programmatically generated 100×100 JPEG. Used for structural tests that do not need a real face. No file dependency — works in CI. |
| `blank_image` | session | Plain white 100×100 JPEG. Used for edge case tests — endpoints should return 200 with empty results, not crash. |
| `bad_file` | session | Plain text bytes. Used to verify that endpoints return 400 when a non-image file is uploaded. |
| `face_image_bytes` | session | Reads `test_assets/Tom_Cruise.jpg`. Used for tests that require real face detection. Skips gracefully if the file is missing. |
| `server_available` | session | Checks the server is reachable before running any tests. |
| `skip_if_server_down` | autouse | Applied to every test automatically. Skips with a clear message if the server is not running, rather than failing with a `ConnectionError`. |

### Test Image Strategy

Tests are split into two categories:

- **Structural tests** — use `sample_image` or `blank_image`. These only check that the response has the right shape and types. No real face needed, always runnable in CI.
- **Face-dependent tests** — use `face_image_bytes`. These verify that the model actually detects a face and returns per-face data. Skip gracefully if `Tom_Cruise.jpg` is missing.

---

## Unit Test Files

Unit tests verify API structure and types only. They do not test whether the model predicts correctly — that belongs in integration tests.

---

### health_unit_tests.py

**Endpoint:** `GET /api/health`

This is the first test in CI/CD — it confirms the server started correctly before anything else runs. If this fails, the problem is the server environment, not the code.

| Test | What it checks |
|---|---|
| `test_health_returns_200` | Server responds with HTTP 200 |
| `test_health_returns_json` | Content-Type is `application/json` |
| `test_health_has_status_key` | Response body has `status` key |
| `test_health_status_is_ok` | `status` value equals `"ok"` |
| `test_health_has_service_key` | Response body has `service` key |
| `test_health_response_time` | Response arrives in under 2 seconds |

---

### face_detection_unit_tests.py

**Endpoint:** `POST /api/detection/detect`

Tests the face detection service which is the entry point of the entire inference pipeline. All downstream services (liveness, emotion, verification) receive crops produced by this service, so correctness here is critical.

**Status code tests** — use `sample_image`:

| Test | What it checks |
|---|---|
| `test_returns_200_on_valid_image` | Valid image returns HTTP 200 |
| `test_returns_400_on_non_image_file` | Text file returns HTTP 400 |
| `test_returns_422_on_no_file` | Missing file returns HTTP 422 |

**Response structure tests** — use `sample_image`:

| Test | What it checks |
|---|---|
| `test_response_is_json` | Content-Type is `application/json` |
| `test_response_has_image_width` | Response has `image_width` key |
| `test_response_has_image_height` | Response has `image_height` key |
| `test_response_has_faces` | Response has `faces` key |
| `test_image_width_is_positive_int` | `image_width` is a positive integer |
| `test_image_height_is_positive_int` | `image_height` is a positive integer |
| `test_faces_is_a_list` | `faces` is a list |

**Edge case tests** — use `blank_image`:

| Test | What it checks |
|---|---|
| `test_blank_image_returns_200` | Blank image does not crash the server |
| `test_blank_image_returns_empty_faces_list` | Blank image returns `faces: []` not `None` |

**Per-face structure tests** — use `face_image_bytes` (skip if missing):

| Test | What it checks |
|---|---|
| `test_face_has_bbox` | Each face has a `bbox` key |
| `test_face_has_detection_confidence` | Each face has `detection_confidence` |
| `test_face_has_crop_width` | Each face has `crop_width` |
| `test_face_has_crop_height` | Each face has `crop_height` |
| `test_bbox_has_x/y/w/h` | Bbox has all four coordinate keys |
| `test_bbox_x/y/w/h_is_normalized` | All bbox values in `[0.0, 1.0]` |
| `test_bbox_x_plus_w_does_not_exceed_1` | Bbox does not overflow image width |
| `test_bbox_y_plus_h_does_not_exceed_1` | Bbox does not overflow image height |
| `test_detection_confidence_is_float` | Confidence is a plain Python float |
| `test_detection_confidence_in_range` | Confidence in `[0.0, 1.0]` |
| `test_crop_width_is_positive_int` | Crop width is a positive integer |
| `test_crop_height_is_positive_int` | Crop height is a positive integer |

---

### emotion_unit_tests.py

**Endpoint:** `POST /api/emotion/`

Tests the emotion recognition service. Unlike detection, emotion runs on any image — it always returns a prediction regardless of whether a face is present — so all tests use `sample_image`.

| Test | What it checks |
|---|---|
| `test_returns_200_on_valid_image` | Valid image returns HTTP 200 |
| `test_returns_400_on_non_image_file` | Text file returns HTTP 400 |
| `test_returns_422_on_no_file` | Missing file returns HTTP 422 |
| `test_response_is_json` | Content-Type is `application/json` |
| `test_response_has_emotion_key` | Response has `emotion` key |
| `test_emotion_has_label` | `emotion` has `label` key |
| `test_emotion_has_confidence` | `emotion` has `confidence` key |
| `test_emotion_label_is_string` | Label is a string |
| `test_emotion_label_is_not_empty` | Label is not an empty string |
| `test_emotion_label_is_valid` | Label is one of the 8 valid emotion classes |
| `test_emotion_confidence_is_float` | Confidence is a plain Python float |
| `test_emotion_confidence_in_range` | Confidence in `[0.0, 1.0]` |
| `test_blank_image_returns_200` | Blank image does not crash the server |
| `test_blank_image_returns_valid_structure` | Blank image still returns valid structure |

Valid emotion labels: `anger`, `contempt`, `disgust`, `fear`, `happy`, `neutral`, `sad`, `surprise`.

---

### anti_spoofing_unit_tests.py

**Endpoint:** `POST /api/anti-spoofing/anti-spoof`

Tests the liveness detection service. Same as emotion — runs on any image and always returns a prediction, so all tests use `sample_image`.

| Test | What it checks |
|---|---|
| `test_returns_200_on_valid_image` | Valid image returns HTTP 200 |
| `test_returns_400_on_non_image_file` | Text file returns HTTP 400 |
| `test_returns_422_on_no_file` | Missing file returns HTTP 422 |
| `test_response_is_json` | Content-Type is `application/json` |
| `test_response_has_anti_spoofing_key` | Response has `anti_spoofing` key |
| `test_anti_spoofing_has_label` | `anti_spoofing` has `label` key |
| `test_anti_spoofing_has_confidence` | `anti_spoofing` has `confidence` key |
| `test_anti_spoofing_label_is_string` | Label is a string |
| `test_anti_spoofing_label_is_not_empty` | Label is not empty |
| `test_anti_spoofing_label_is_valid` | Label is `"real"` or `"spoof"` |
| `test_anti_spoofing_confidence_is_float` | Confidence is a plain Python float |
| `test_anti_spoofing_confidence_in_range` | Confidence in `[0.0, 1.0]` |
| `test_blank_image_returns_200` | Blank image does not crash the server |
| `test_blank_image_returns_valid_structure` | Blank image still returns valid structure |

---

### verification_unit_tests.py

**Endpoints:** `POST /api/verification/register` and `POST /api/verification/verify`

Tests the face verification service. Both register and verify require a real face image to extract a valid ArcFace embedding, so all face-dependent tests use `face_image_bytes`.

> **Known limitation:** Register and verify use separate service instances in the current codebase. A face registered via `/register` is not visible to `/verify` through the pipeline. This is a known shared-state bug tracked separately. Tests are written around current behaviour.

**Register tests:**

| Test | What it checks |
|---|---|
| `test_register_returns_200_on_valid_image` | Valid image returns HTTP 200 |
| `test_register_returns_400_on_non_image_file` | Text file returns HTTP 400 |
| `test_register_returns_422_on_missing_person_id` | Missing `person_id` returns HTTP 422 |
| `test_register_returns_422_on_no_file` | Missing file returns HTTP 422 |
| `test_register_response_is_json` | Content-Type is `application/json` |
| `test_register_has_person_id` | Response has `person_id` key |
| `test_register_has_status` | Response has `status` key |
| `test_register_has_message` | Response has `message` key |
| `test_register_person_id_matches_input` | `person_id` in response matches input |
| `test_register_status_is_valid` | Status is `"registered"` or `"updated"` |
| `test_register_twice_returns_updated` | Second registration of same ID returns `"updated"` |
| `test_register_message_is_string` | Message is a non-empty string |

**Verify tests:**

| Test | What it checks |
|---|---|
| `test_verify_returns_200_on_valid_image` | Valid image returns HTTP 200 |
| `test_verify_returns_400_on_non_image_file` | Text file returns HTTP 400 |
| `test_verify_returns_422_on_no_file` | Missing file returns HTTP 422 |
| `test_verify_response_is_json` | Content-Type is `application/json` |
| `test_verify_has_recognition_key` | Response has `recognition` key |
| `test_verify_recognition_has_label` | `recognition` has `label` key |
| `test_verify_recognition_has_confidence` | `recognition` has `confidence` key |
| `test_verify_recognition_has_matched` | `recognition` has `matched` key |
| `test_verify_label_is_string` | Label is a string |
| `test_verify_label_is_not_empty` | Label is not empty |
| `test_verify_confidence_is_float` | Confidence is a plain Python float |
| `test_verify_confidence_in_range` | Confidence in `[0.0, 1.0]` |
| `test_verify_matched_is_bool` | Matched is a boolean |
| `test_verify_matched_is_bool_after_registration` | Matched is a boolean after registration tests run |

---

### pipeline_unit_tests.py

**Endpoint:** `POST /api/pipeline/frame`

Tests the full inference pipeline which runs all four services in sequence: detection → anti-spoofing → emotion → verification. This is the primary production endpoint. Timeout is set to 60 seconds since all models run sequentially.

**Status code tests** — use `sample_image`:

| Test | What it checks |
|---|---|
| `test_returns_200_on_valid_image` | Valid image returns HTTP 200 |
| `test_returns_400_on_non_image_file` | Text file returns HTTP 400 |
| `test_returns_422_on_no_file` | Missing file returns HTTP 422 |

**Response structure tests** — use `sample_image`:

| Test | What it checks |
|---|---|
| `test_response_is_json` | Content-Type is `application/json` |
| `test_response_has_image_width` | Response has `image_width` |
| `test_response_has_image_height` | Response has `image_height` |
| `test_response_has_faces` | Response has `faces` |
| `test_image_width_is_positive_int` | `image_width` is a positive integer |
| `test_image_height_is_positive_int` | `image_height` is a positive integer |
| `test_faces_is_a_list` | `faces` is a list |

**Edge case tests** — use `blank_image`:

| Test | What it checks |
|---|---|
| `test_blank_image_returns_200` | Blank image does not crash the pipeline |
| `test_blank_image_returns_empty_faces` | Blank image returns `faces: []` |

**Per-face structure tests** — use `face_image_bytes` (skip if missing):

| Test | What it checks |
|---|---|
| `test_face_has_face_key` | Each entry has `face` key |
| `test_face_has_emotion_key` | Each entry has `emotion` key |
| `test_face_has_anti_spoofing_key` | Each entry has `anti_spoofing` key |
| `test_face_has_recognition_key` | Each entry has `recognition` key |
| `test_face_bbox_has_x/y/w/h` | Bbox has all four coordinate keys |
| `test_face_bbox_values_normalized` | All bbox values in `[0.0, 1.0]` |
| `test_detection_confidence_is_float` | Detection confidence is a float |
| `test_detection_confidence_in_range` | Detection confidence in `[0.0, 1.0]` |
| `test_emotion_label_is_valid` | Emotion label is a valid class |
| `test_emotion_confidence_in_range` | Emotion confidence in `[0.0, 1.0]` |
| `test_anti_spoofing_label_is_valid` | Anti-spoofing label is `"real"` or `"spoof"` |
| `test_anti_spoofing_confidence_in_range` | Anti-spoofing confidence in `[0.0, 1.0]` |
| `test_recognition_label_is_string` | Recognition label is a string |
| `test_recognition_confidence_in_range` | Recognition confidence in `[0.0, 1.0]` |
| `test_recognition_matched_is_bool` | Matched is a boolean |

---

## Integration Test Folder

`API_endpoint_integration_test/` is currently empty. Integration tests will be added here to verify model behaviour — for example:

- Given a clearly happy face → emotion label should be `"happy"`
- Given a printed photo → anti-spoofing should return `"spoof"`
- Given a registered face → verification should return `matched: True`

These require labelled images in `test_assets/` and will be added as the project matures.

---

## CI/CD Readiness

The test infrastructure is ready for GitHub Actions. When introduced, the pipeline will:

1. Install dependencies via `pip install -r backend/requirements-dev.txt`
2. Start the uvicorn server
3. Wait for `GET /api/health` to return 200 — confirmed by `health_unit_tests.py` running first
4. Run `pytest -v`
5. Report results — all tests must pass for a PR to merge

The `skip_if_server_down` fixture in `conftest.py` ensures that if the server fails to start, all tests skip with a clear message rather than failing with confusing connection errors.

---

## Known Issues

| Issue | Location | Status |
|---|---|---|
| Separate `VerificationService` instances in route and pipeline | `routes_verification.py`, `inference_service.py` | Known, tracked |
| `arcface.onnx` required for verification and pipeline face-dependent tests | `weights/` | Resolved — file added |
| CUDA 13.1 not supported by `onnxruntime-gpu` | Environment | Running on CPU fallback, GPU fix deferred |
