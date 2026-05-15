import os
from datetime import datetime

import cv2 as cv


# Run face detection every 15 frames
FRAME_INTERVAL = 15

# Folder to save detected face crops
FACE_IMG_DIR = "./Test_Face_Detect/Frame_IMGs"
os.makedirs(FACE_IMG_DIR, exist_ok=True)

# OpenCV face detector
FACE_CASCADE_PATH = cv.data.haarcascades + "haarcascade_frontalface_default.xml"
face_cascade = cv.CascadeClassifier(FACE_CASCADE_PATH)

if face_cascade.empty():
    raise RuntimeError("Could not load Haar Cascade face detector.")


def detect_faces(frame):
    """
    Detects faces in a webcam frame.

    Returns a list of bounding boxes:
    (x, y, w, h)
    """

    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(40, 40)
    )

    return faces


def save_face_images(frame, faces, frame_count):
    """
    Saves cropped face images from the frame.
    """

    if len(faces) == 0:
        return

    frame_height, frame_width = frame.shape[:2]
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    for index, (x, y, w, h) in enumerate(faces, start=1):

        # Keep bbox inside frame boundaries
        x1 = max(0, x)
        y1 = max(0, y)
        x2 = min(frame_width, x + w)
        y2 = min(frame_height, y + h)

        if x2 <= x1 or y2 <= y1:
            continue

        # Crop face from frame
        face_crop = frame[y1:y2, x1:x2]

        file_name = f"frame_{frame_count:06d}_{timestamp}_face_{index}.jpg"
        output_path = os.path.join(FACE_IMG_DIR, file_name)

        cv.imwrite(output_path, face_crop)

        print(f"Saved face image: {output_path}")


# Open webcam
cap = cv.VideoCapture(0)

if not cap.isOpened():
    raise RuntimeError("Could not open webcam.")

frame_count = 0
current_faces = []

while True:

    ret, frame = cap.read()

    if not ret:
        break

    # Detect and save faces every 15 frames
    if frame_count % FRAME_INTERVAL == 0:
        current_faces = detect_faces(frame)

        save_face_images(
            frame=frame,
            faces=current_faces,
            frame_count=frame_count
        )

    # Draw face bounding boxes on the live webcam display
    for (x, y, w, h) in current_faces:
        cv.rectangle(
            frame,
            (x, y),
            (x + w, y + h),
            (0, 255, 0),
            2
        )

        cv.putText(
            frame,
            "Face",
            (x, y - 10 if y > 20 else y + 20),
            cv.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2
        )

    cv.imshow("Face Detection", frame)

    frame_count += 1

    if cv.waitKey(1) == ord("q"):
        break


cap.release()
cv.destroyAllWindows()