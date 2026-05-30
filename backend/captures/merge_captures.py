import json
import os
from pathlib import Path
from PIL import Image
import random

#Check for CAPTURES_DIR (for devops) otherwise use backend/captures/screenshots for local
screenshots_dir = Path(os.environ.get("CAPTURES_DIR", "backend/captures/screenshots"))

#Anti Spoofing Dir Paths
train_fasd_path = Path(os.environ.get("TRAIN_FASD_PATH", "LCC_FASD/LCC_FASD_training"))
test_fasd_path = Path(os.environ.get("TEST_FASD_PATH", "LCC_FASD/LCC_FASD_evaluation"))

#Emotion Dect Paths
train_affectnet_path = Path(os.environ.get("TRAIN_AFFECTNET_PATH", "AffectNet/Train"))
test_affectnet_path = Path(os.environ.get("TEST_AFFECTNET_PATH", "AffectNet/Test"))

#Recognition Path
recognition_path = Path(os.environ.get("RECOGNITION_PATH", "recognition_data"))

#Confidence Thresholds
min_detection_confidence = 0.7
min_label_confidence = 0.7


#Read file path and open JSON file + Return data
def read_path(file_path):
    with open(file_path, "r") as file:
        data = json.load(file)

    return data

#Get the face number (if more than 1 face in JSON)
def get_face(data, number):
    face_number = data["faces"][number]
    
    return face_number

#Get all properties for given face
def get_properties(data, face_number):
    image_width = data["image_width"]
    image_height = data["image_height"]

    bbox = face_number["face"]["bbox"]
    detection_confidence = face_number["face"]["detection_confidence"]

    emotion = face_number["emotion"]
    anti_spoofing = face_number["anti_spoofing"]
    recognition = face_number["recognition"]

    return image_width, image_height, bbox, detection_confidence, emotion, anti_spoofing, recognition

#Build boudning box (face)
def build_bounding_box(bbox, image_width, image_height):
    left = bbox["x"] * image_width
    top = bbox["y"] * image_height
    right = (bbox["x"] + bbox["w"]) * image_width
    bottom = (bbox["y"] + bbox["h"]) * image_height

    return int(left), int(top), int(right), int(bottom)

#Crop face from img using bounding box
def crop_face(image, bbox, image_width, image_height):
    left, top, right, bottom = build_bounding_box(bbox, image_width, image_height)

    return image.crop((left, top, right, bottom))

#Randomly split training and testing Anti Spoof + Emotion
def save_split():
    if random.random() < 0.7:
        return "train"
    else:
        return "test"

#Randomly split recognition 
def save_recognition_split():
    value = random.random()

    if value < 0.33:
        return "train"
    elif value < 0.67:
        return "val"
    else:
        return "test"


def main():
    #Loop through all json files
    for file in screenshots_dir.glob("*.json"):
        data = read_path(file)

        #Get corresponding jpg file
        image_file = file.with_suffix(".jpg")

        #Track if the img was saved
        saved_true = False

        #Loop through faces in json
        for i in range(len(data["faces"])):
            face = get_face(data, i)

            image_width, image_height, bbox, detection_confidence, emotion, anti_spoofing, recognition = get_properties(data, face)

            #Check if face was actually detected with min confidence
            if detection_confidence < min_detection_confidence:
                print(f"Skipped {file.name} Face {i + 1}: low detection confidence")
                continue
            
            #Open img file and crop out face
            image = Image.open(image_file)
            cropped_face = crop_face(image, bbox, image_width, image_height)

            #If confidence > thresehold save to anti-spoofing training path (either train or test)
            if anti_spoofing["confidence"] >= min_label_confidence:
                spoofing_label = anti_spoofing["label"]

                split = save_split()
                if split == "train":
                    output_path = train_fasd_path / spoofing_label
                else:
                    output_path = test_fasd_path / spoofing_label

                output_file = output_path / f"{file.stem}_{i + 1}.jpg"

                cropped_face.save(output_file)

                saved_true = True

                print(f"Saved {file.name} Face {i + 1} to {output_file} Confidence: {anti_spoofing['confidence']}")

            #If confidence > thresehold save to emotion training path (either train or test)
            if emotion["confidence"] >= min_label_confidence:
                emotion_label = emotion["label"]

                split = save_split()
                if split == "train":
                    output_path = train_affectnet_path / emotion_label
                else:
                    output_path = test_affectnet_path / emotion_label

                output_file = output_path / f"{file.stem}_{i + 1}.jpg"

                cropped_face.save(output_file)

                saved_true = True

                print(f"Saved {file.name} Face {i + 1} to {output_file} Confidence: {emotion['confidence']}")

            #If confidence > threshold, check name assigned to face and is match then store in training, testing or val path under identitity
            if recognition["confidence"] >= 0.5:
                if recognition["label"] == "unknown" or recognition["matched"] is False:
                    continue

                identity = recognition["label"]
                split = save_recognition_split()
                
                if split == "train":
                    output_path = recognition_path / "train_data" / identity
                    
                elif split == "val":
                    output_path = recognition_path / "val_data" / identity
                else:
                    output_path = recognition_path / "test_data" / identity

                output_path.mkdir(parents=True, exist_ok=True)

                output_file = output_path / f"{file.stem}_{i + 1}.jpg"

                cropped_face.save(output_file)

                saved_true = True

                print(f"Saved {file.name} Face {i + 1} to {output_file} Confidence: {recognition['confidence']}")
    
        #Remove raw files if saved somewhere
        if saved_true:
            file.unlink()
            image_file.unlink()

            print(f"Deleted {file.name} and {image_file.name}")

if __name__ == "__main__":
    main()