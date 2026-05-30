import json
import time
from datetime import datetime

from app.config import settings

#SETTINGS
min_interval_secs = 10
only_if_faces = True

last_save_at = 0.0

#Save Capture Function
def save(image_bytes, payload):
    global last_save_at

    #Settings enabled
    if not settings.captures_enabled:
        return

    #Check for Faces in Json file
    if only_if_faces and not payload.get("faces"):
        return
    
    #Stopwatch 
    current_time = time.monotonic()

    #Check current time is > 10 seconds to save - Update last save
    if current_time - last_save_at < min_interval_secs:
        return

    last_save_at = current_time

    #Captures/Screenshot dir
    output_dir = settings.captures_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    #Use timestamp as name
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")

    #Save Img (Bytes) and Json (Payload)
    image_path = output_dir / f"{timestamp}.jpg"
    json_path = output_dir / f"{timestamp}.json"

    image_path.write_bytes(image_bytes)

    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    