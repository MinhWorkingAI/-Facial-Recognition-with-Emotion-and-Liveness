import argparse
import mimetypes
import sys
from pathlib import Path

import requests


def main() -> int:
    parser = argparse.ArgumentParser(description="Send a single image to the frame API")
    parser.add_argument("--image", required=True, help="Path to an image file")
    parser.add_argument("--url", default="http://127.0.0.1:8000/api/pipeline/frame")
    args = parser.parse_args()

    image_path = Path(args.image)
    if not image_path.is_file():
        print("Image file not found.")
        return 1

    mime_type, _ = mimetypes.guess_type(image_path.name)
    if not mime_type:
        mime_type = "application/octet-stream"

    with image_path.open("rb") as handle:
        files = {"file": (image_path.name, handle, mime_type)}
        response = requests.post(args.url, files=files, timeout=30)

    print("Status:", response.status_code)
    try:
        print(response.json())
    except ValueError:
        print(response.text)

    return 0


if __name__ == "__main__":
    sys.exit(main())
