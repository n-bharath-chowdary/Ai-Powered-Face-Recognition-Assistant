import cv2
import face_recognition
import pickle
import os


def train_face_model(dataset_dir="facedata", output_file="encodings.pickle"):
    known_encodings = []
    known_names = []

    for root, dirs, files in os.walk(dataset_dir):
        for file in files:
            if file.endswith(".jpg") or file.endswith(".png"):
                path = os.path.join(root, file)
                name = os.path.basename(root)

                image = cv2.imread(path)
                rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

                boxes = face_recognition.face_locations(rgb, model="hog")
                encodings = face_recognition.face_encodings(rgb, boxes)

                for encoding in encodings:
                    known_encodings.append(encoding)
                    known_names.append(name)

    data = {"encodings": known_encodings, "names": known_names}
    with open(output_file, "wb") as f:
        pickle.dump(data, f)

    print("[INFO] Training completed.")


if __name__ == "__main__":
    train_face_model()
