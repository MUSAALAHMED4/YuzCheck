# face_recognition_class.py

import os
import cv2
import torch
from face_recognition.arcface.model import iresnet_inference
from face_recognition.arcface.utils import compare_encodings, read_features
from face_alignment.alignment import norm_crop
import numpy as np

# Ensure device consistency (CPU or GPU)
device = 'cuda' if torch.cuda.is_available() else 'cpu'


class FaceRecognitionClass:
    def __init__(self, model_name="r100", model_path="face_recognition/arcface/weights/arcface_r100.pth",
                 feature_path="./datasets/face_features/feature", input_size=112, min_score=0.6):
        """ Initialize the FaceRecognitionClass with the model and feature paths """
        self.device = device
        self.input_size = input_size
        self.min_score = min_score

        # Initialize ArcFace recognizer
        self.recognizer = iresnet_inference(model_name=model_name, path=model_path, device=self.device)

        # Load precomputed face features and names
        self.images_names, self.images_embs = read_features(feature_path=feature_path)

    def preprocess_face(self, face_img):
        """ Preprocess the face image to the format expected by the ArcFace model """
        if face_img is None or face_img.size == 0:
            raise ValueError("Invalid face image for preprocessing.")

        face_img = cv2.resize(face_img, (self.input_size, self.input_size))
        face_img = face_img.astype(np.float32) / 255.0
        face_img = (face_img - 0.5) / 0.5  # Normalize to [-1, 1]
        face_img = torch.from_numpy(face_img).permute(2, 0, 1).unsqueeze(0)  # Add batch and channel dimensions (CHW)

        return face_img.to(self.device)  # Ensure the face image tensor is on the same device as the model (GPU or CPU)

    def get_feature(self, face_img):
        """ Extract feature from the face image """
        face_tensor = self.preprocess_face(face_img)
        with torch.no_grad():
            embedding = self.recognizer(face_tensor)  # Pass the tensor through the model
            embedding = torch.nn.functional.normalize(embedding, p=2, dim=1)  # L2 normalize the embedding

        return embedding.cpu().numpy()  # Return embedding as a CPU tensor (numpy array)

    def recognize_face(self, face_image):
        try:
            query_emb = self.get_feature(face_image)
            score, id_min = compare_encodings(query_emb, self.images_embs)

            # `self.images_names`
            name = self.images_names[id_min] if id_min is not None else "UNKNOWN"
            score = score[0] if score is not None else 0.0

            if not name or name.strip() == "":
                name = "UNKNOWN"

            return score, name
        except Exception as e:
            print(f"⚠️ Recognition failed: {str(e)}")
            return None, "UNKNOWN"

    def process_folder(self, pimages_dir="detected_faces"):
        """ Process a folder of face images and recognize faces """
        # Check if folder exists
        if not os.path.exists(pimages_dir):
            raise FileNotFoundError(f"The folder '{pimages_dir}' does not exist.")

        # Loop over all saved face images
        for img_name in os.listdir(pimages_dir):
            img_path = os.path.join(pimages_dir, img_name)
            face_img = cv2.imread(img_path)

            if face_img is None or face_img.size == 0:
                print(f"Invalid image {img_name}")
                continue

            # Recognize the face
            score, name = self.recognize_face(face_img)

            if score is not None and score >= self.min_score:
                print(f"Recognized {name} with score {score:.2f} in {img_name}")
            else:
                print(f"Unknown face in {img_name}")


# Example usage
if __name__ == '__main__':
    face_recognizer = FaceRecognitionClass(
        model_name="r100",
        model_path="face_recognition/arcface/weights/arcface_r100.pth",
        feature_path="./datasets/face_features/feature",
        input_size=112,
        min_score=0.4
    )

    # Process the folder with detected face images
    face_recognizer.process_folder(pimages_dir="detected_faces")
