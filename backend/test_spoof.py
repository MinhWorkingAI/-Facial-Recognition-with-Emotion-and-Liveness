import argparse
from pathlib import Path

import requests


def main() -> None:
    parser = argparse.ArgumentParser(description="Test /api/anti-spoofing/anti-spoof endpoint.")
    parser.add_argument("--image", required=True, help="Path to an image file.")
    parser.add_argument(
        "--url",
        default="http://127.0.0.1:8000/api/anti-spoofing/anti-spoof",
        help="Anti-spoofing endpoint URL.",
    )
    args = parser.parse_args()

    image_path = Path(args.image)
    if not image_path.is_file():
        raise FileNotFoundError(f"Image not found: {image_path}")

    with image_path.open("rb") as handle:
        files = {"file": (image_path.name, handle, "image/jpeg")}
        response = requests.post(args.url, files=files, timeout=30)

    print(f"Status: {response.status_code}")
    try:
        print(response.json())
    except ValueError:
        print(response.text)


if __name__ == "__main__":
    main()
