import os
import shutil
import cv2
import numpy as np
import torch
from torchvision import transforms

from face_detection.scrfd.detector import SCRFD
from face_recognition.arcface.model import iresnet_inference
from face_recognition.arcface.utils import read_features

# CUDA kullanılıyorsa kontrol et
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Yüz algılama modelini yükle
detector = SCRFD(model_file="face_detection/scrfd/weights/scrfd_2.5g_bnkps.onnx")

# Yüz tanıma modelini yükle
recognizer = iresnet_inference(
    model_name="r100", path="face_recognition/arcface/weights/arcface_r100.pth", device=device
)


@torch.no_grad()
def get_feature(face_image):
    """
    Yüz tanıma modeli kullanarak özellikleri (features) çıkar.
    """
    face_preprocess = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Resize((112, 112)),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ]
    )

    face_image = cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB)
    face_image = face_preprocess(face_image).unsqueeze(0).to(device)

    emb_img_face = recognizer(face_image)[0].cpu().numpy()
    return emb_img_face / np.linalg.norm(emb_img_face)


def add_all_faces():
    """
  `original/` klasöründeki tüm resimleri işleyerek veritabanına ekler.
    """
    # Varsayılan yollar
    backup_dir = "./datasets/backup"
    add_persons_dir = "./datasets/new_persons/original"
    faces_save_dir = "./datasets/data/"
    features_path = "./datasets/face_features/feature"

    images_name = []
    images_emb = []

    #  Gerekli klasörlerin mevcut olduğundan emin ol
    os.makedirs(backup_dir, exist_ok=True)
    os.makedirs(faces_save_dir, exist_ok=True)
    os.makedirs(os.path.dirname(features_path), exist_ok=True)

    #  `original/` klasöründeki her bir resmi işle
    for image_name in os.listdir(add_persons_dir):
        if not image_name.lower().endswith(("png", "jpg", "jpeg")):
            continue  # 🔹 Resim olmayan dosyaları atla

        image_path = os.path.join(add_persons_dir, image_name)
        input_image = cv2.imread(image_path)

        if input_image is None:
            print(f"️ Resim yüklenemedi: {image_path}")
            continue

        #  Kişi adını dosya adından çıkar (uzantıdan önceki kısım)
        person_name = os.path.splitext(image_name)[0]

        #  Yüz algılama işlemi
        bboxes, landmarks = detector.detect(image=input_image)

        for i in range(len(bboxes)):
            x1, y1, x2, y2, score = map(int, bboxes[i])

            #  Yüzün resim sınırları içinde olup olmadığını kontrol et
            if x1 < 0 or y1 < 0 or x2 > input_image.shape[1] or y2 > input_image.shape[0]:
                print(f" Yüz atlandı ({x1}, {y1}, {x2}, {y2}) - sınırların dışında!")
                continue

            #  Yüz bölgesini kes
            face_image = input_image[y1:y2, x1:x2]
            if face_image.size == 0:
                print(" Boş yüz resmi - atlanıyor!")
                continue

            #  Kesilmiş yüzün kaydedileceği yolu belirle
            person_face_path = os.path.join(faces_save_dir, person_name)
            os.makedirs(person_face_path, exist_ok=True)

            face_count = len(os.listdir(person_face_path))
            path_save_face = os.path.join(person_face_path, f"{face_count}.jpg")
            cv2.imwrite(path_save_face, face_image)

            #  Özellikleri çıkar ve listeye ekle
            images_emb.append(get_feature(face_image))
            images_name.append(person_name)

    if not images_emb:
        print(" Eklemek için yüz bulunamadı!")
        return

    #  Listeleri dizilere çevir
    images_emb = np.array(images_emb)
    images_name = np.array(images_name)

    #  Önceden kaydedilmiş özellikleri yükle (varsa)
    features = read_features(features_path)
    if features is not None:
        old_images_name, old_images_emb = features
        images_name = np.hstack((old_images_name, images_name))
        images_emb = np.vstack((old_images_emb, images_emb))
        print(" Veritabanı yeni özelliklerle güncellendi!")

    # Güncellenmiş özellikleri kaydet
    np.savez_compressed(features_path, images_name=images_name, images_emb=images_emb)
    print("Tüm özellikler veritabanına kaydedildi!")

    #  Orijinal resimleri `backup/` klasörüne taşı
    shutil.move(add_persons_dir, os.path.join(backup_dir, "original"))
    os.makedirs(add_persons_dir, exist_ok=True)  # `original/` klasörünü boş olarak yeniden oluştur

    print(" Tüm kişiler başarıyla eklendi!")


if __name__ == "__main__":
    add_all_faces()