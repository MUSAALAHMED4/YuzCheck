import pandas as pd
import os
from datetime import datetime, timedelta
import cv2
import numpy as np
import torch
import argparse
import onnxruntime as ort
from face_recognition.arcface.model import iresnet_inference
from face_recognition.arcface.utils import compare_encodings, read_features
from face_alignment.alignment import norm_crop
import math

attendance_file = "attendance_log.xlsx"
last_logged_times = {}

class YOLOv8_face:
    def __init__(self, path, conf_thres=0.2, iou_thres=0.5):
        self.conf_threshold = conf_thres
        self.iou_threshold = iou_thres
        self.class_names = ['face']
        self.num_classes = len(self.class_names)

        # Load model and move to GPU
        self.model = ort.InferenceSession(path, providers=['CUDAExecutionProvider'] if torch.cuda.is_available() else ['CPUExecutionProvider'])

        self.input_height = 640
        self.input_width = 640
        self.reg_max = 16

        self.project = np.arange(self.reg_max)
        self.strides = (8, 16, 32)
        self.feats_hw = [(math.ceil(self.input_height / self.strides[i]), math.ceil(self.input_width / self.strides[i]))
                         for i in range(len(self.strides))]
        self.anchors = self.make_anchors(self.feats_hw)



    def make_anchors(self, feats_hw, grid_cell_offset=0.5):
        anchor_points = {}
        for i, stride in enumerate(self.strides):
            h, w = feats_hw[i]
            x = np.arange(0, w) + grid_cell_offset  # shift x
            y = np.arange(0, h) + grid_cell_offset  # shift y
            sx, sy = np.meshgrid(x, y)
            anchor_points[stride] = np.stack((sx, sy), axis=-1).reshape(-1, 2)
        return anchor_points

    def softmax(self, x, axis=1):
        x_exp = np.exp(x)
        x_sum = np.sum(x_exp, axis=axis, keepdims=True)
        s = x_exp / x_sum
        return s

    def resize_image(self, srcimg, keep_ratio=True):
        top, left, newh, neww = 0, 0, self.input_width, self.input_height
        if keep_ratio and srcimg.shape[0] != srcimg.shape[1]:
            hw_scale = srcimg.shape[0] / srcimg.shape[1]
            if hw_scale > 1:
                newh, neww = self.input_height, int(self.input_width / hw_scale)
                img = cv2.resize(srcimg, (neww, newh), interpolation=cv2.INTER_AREA)
                left = int((self.input_width - neww) * 0.5)
                img = cv2.copyMakeBorder(img, 0, 0, left, self.input_width - neww - left, cv2.BORDER_CONSTANT,
                                         value=(0, 0, 0))  # add border
            else:
                newh, neww = int(self.input_height * hw_scale), self.input_width
                img = cv2.resize(srcimg, (neww, newh), interpolation=cv2.INTER_AREA)
                top = int((self.input_height - newh) * 0.5)
                img = cv2.copyMakeBorder(img, top, self.input_height - newh - top, 0, 0, cv2.BORDER_CONSTANT,
                                         value=(0, 0, 0))
        else:
            img = cv2.resize(srcimg, (self.input_width, self.input_height), interpolation=cv2.INTER_AREA)
        return img, newh, neww, top, left

    def detect(self, srcimg):
        input_img, newh, neww, padh, padw = self.resize_image(cv2.cvtColor(srcimg, cv2.COLOR_BGR2RGB))
        scale_h, scale_w = srcimg.shape[0] / newh, srcimg.shape[1] / neww
        input_img = input_img.astype(np.float32) / 255.0
        input_img = np.transpose(input_img, (2, 0, 1))  # Change to CHW format

        # Run inference using ONNX Runtime
        blob = np.expand_dims(input_img, axis=0)  # Add batch dimension
        # Use the correct input name
        outputs = self.model.run(None, {self.model.get_inputs()[0].name: blob})

        det_bboxes, det_conf, det_classid, landmarks = self.post_process(outputs, scale_h, scale_w, padh, padw)
        return det_bboxes, det_conf, det_classid, landmarks

    def post_process(self, preds, scale_h, scale_w, padh, padw):
        bboxes, scores, landmarks = [], [], []
        for i, pred in enumerate(preds):
            stride = int(self.input_height / pred.shape[2])
            pred = pred.transpose((0, 2, 3, 1))

            box = pred[..., :self.reg_max * 4]
            cls = 1 / (1 + np.exp(-pred[..., self.reg_max * 4:-15])).reshape((-1, 1))
            kpts = pred[..., -15:].reshape((-1, 15))

            tmp = box.reshape(-1, 4, self.reg_max)
            bbox_pred = self.softmax(tmp, axis=-1)
            bbox_pred = np.dot(bbox_pred, self.project).reshape((-1, 4))

            bbox = self.distance2bbox(self.anchors[stride], bbox_pred,
                                      max_shape=(self.input_height, self.input_width)) * stride
            kpts[:, 0::3] = (kpts[:, 0::3] * 2.0 + (self.anchors[stride][:, 0].reshape((-1, 1)) - 0.5)) * stride
            kpts[:, 1::3] = (kpts[:, 1::3] * 2.0 + (self.anchors[stride][:, 1].reshape((-1, 1)) - 0.5)) * stride
            kpts[:, 2::3] = 1 / (1 + np.exp(-kpts[:, 2::3]))

            bbox -= np.array([[padw, padh, padw, padh]])
            bbox *= np.array([[scale_w, scale_h, scale_w, scale_h]])
            kpts -= np.tile(np.array([padw, padh, 0]), 5).reshape((1, 15))
            kpts *= np.tile(np.array([scale_w, scale_h, 1]), 5).reshape((1, 15))

            bboxes.append(bbox)
            scores.append(cls)
            landmarks.append(kpts)

        bboxes = np.concatenate(bboxes, axis=0)
        scores = np.concatenate(scores, axis=0)
        landmarks = np.concatenate(landmarks, axis=0)

        bboxes_wh = bboxes.copy()
        bboxes_wh[:, 2:4] = bboxes[:, 2:4] - bboxes[:, 0:2]  # Convert to xywh
        classIds = np.argmax(scores, axis=1)
        confidences = np.max(scores, axis=1)  # Maximum class confidence

        mask = confidences > self.conf_threshold
        bboxes_wh = bboxes_wh[mask]
        confidences = confidences[mask]
        classIds = classIds[mask]
        landmarks = landmarks[mask]

        indices = cv2.dnn.NMSBoxes(bboxes_wh.tolist(), confidences.tolist(), self.conf_threshold,
                                   self.iou_threshold)

        if len(indices) > 0:
            indices = indices.flatten()
            mlvl_bboxes = bboxes_wh[indices]
            confidences = confidences[indices]
            classIds = classIds[indices]
            landmarks = landmarks[indices]
            return mlvl_bboxes, confidences, classIds, landmarks
        else:
            return np.array([]), np.array([]), np.array([]), np.array([])

    def distance2bbox(self, points, distance, max_shape=None):
        x1 = points[:, 0] - distance[:, 0]
        y1 = points[:, 1] - distance[:, 1]
        x2 = points[:, 0] + distance[:, 2]
        y2 = points[:, 1] + distance[:, 3]
        if max_shape is not None:
            x1 = np.clip(x1, 0, max_shape[1])
            y1 = np.clip(y1, 0, max_shape[0])
            x2 = np.clip(x2, 0, max_shape[1])
            y2 = np.clip(y2, 0, max_shape[0])
        return np.stack([x1, y1, x2, y2], axis=-1)

    def draw_detections(self, image, boxes, scores, kpts):
        for box, kp in zip(boxes, kpts):
            x, y, w, h = box.astype(int)
            # Draw rectangle around face
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 255), thickness=2)
            # Draw keypoints
            for i in range(5):
                cv2.circle(image, (int(kp[i * 3]), int(kp[i * 3 + 1])), 4, (0, 255, 0), thickness=-1)
        return image

def log_to_excel(name):
    """تسجيل الحضور في ملف Excel مع منع التكرار خلال دقيقة واحدة."""
    current_time = datetime.now()

    # ✅ منع التكرار خلال دقيقة واحدة
    if name in last_logged_times:
        last_logged = last_logged_times[name]
        if current_time - last_logged < timedelta(minutes=1):
            return False

    # ✅ تحديث آخر وقت تسجيل للحضور
    last_logged_times[name] = current_time
    timestamp = current_time.strftime('%Y-%m-%d %H:%M:%S')

    # ✅ إنشاء سجل جديد للحضور
    data = {'Name': [name], 'Timestamp': [timestamp]}

    try:
        if os.path.exists(attendance_file):
            existing_df = pd.read_excel(attendance_file)
            new_df = pd.DataFrame(data)
            updated_df = pd.concat([existing_df, new_df], ignore_index=True)
        else:
            updated_df = pd.DataFrame(data)

        # ✅ حفظ الملف بعد التحديث
        updated_df.to_excel(attendance_file, index=False)

        print(f"\n📌 تم تسجيل حضور:\n👤 الاسم: {name}\n🕒 الوقت: {timestamp}\n")
        return True

    except Exception as e:
        print(f"❌ خطأ أثناء تسجيل الحضور في ملف Excel: {e}")
        return False


def preprocess_face(face_img, input_size=112):
    """ Preprocess the face image to the format expected by the ArcFace model """
    if face_img is None or face_img.size == 0:
        raise ValueError("Invalid face image for preprocessing.")

    face_img = cv2.resize(face_img, (input_size, input_size))
    face_img = face_img.astype(np.float32) / 255.0
    face_img = (face_img - 0.5) / 0.5  # Normalize to [-1, 1]
    face_img = np.transpose(face_img, (2, 0, 1))  # Convert to CHW format
    face_img = torch.from_numpy(face_img).unsqueeze(0)  # Add batch dimension
    return face_img.to(device)

def get_feature(face_img, recognizer):
    """ Extract feature from the face image """
    face_tensor = preprocess_face(face_img)
    with torch.no_grad():
        embedding = recognizer(face_tensor)
        embedding = torch.nn.functional.normalize(embedding, p=2, dim=1)  # L2 normalize the embedding
    return embedding.cpu().numpy()

def recognition(face_image, images_embs, images_names):
    """ Recognize a face image """
    try:
        query_emb = get_feature(face_image, recognizer)
        score, id_min = compare_encodings(query_emb, images_embs)
        name = images_names[id_min]
        score = score[0]
        return score, name
    except Exception as e:
        print(f"Recognition failed: {str(e)}")
        return None, "UNKNOWN"

def finalize_attendance():
    """تحديث ملف الحضور والغياب بعد انتهاء الجلسة."""
    student_list_file = "ogrenci_listesi_birlesik.xlsx"

    try:
        # ✅ تحميل قائمة الطلاب الكاملة من ملف الإكسل
        df = pd.read_excel(student_list_file)
        if 'Student Name' not in df.columns:
            print(f"❌ خطأ: لم يتم العثور على العمود 'Student Name' في {student_list_file}")
            return

        expected_names = df['Student Name'].dropna().astype(str).tolist()
        print(f"📌 قائمة الطلاب المسجلين: {expected_names}")

    except Exception as e:
        print(f"❌ خطأ أثناء قراءة قائمة الطلاب: {e}")
        return

    # ✅ استخراج أسماء الحضور الفعلي
    recognized_names = list(last_logged_times.keys())

    # ✅ مقارنة الحضور مع القائمة الكاملة لتحديد الغياب
    absent_names = list(set(expected_names) - set(recognized_names))

    print("\n✅ قائمة الحضور:")
    for name in recognized_names:
        print(f"🟢 {name}")

    print("\n❌ قائمة الغياب:")
    for name in absent_names:
        print(f"🔴 {name}")

    # ✅ تسجيل الحضور والغياب في ملف الإكسل
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    attendance_data = pd.DataFrame({
        'Timestamp': [timestamp],
        'Present Students': [", ".join(recognized_names) if recognized_names else "None"],
        'Absent Students': [", ".join(absent_names) if absent_names else "None"]
    })

    try:
        if os.path.exists(attendance_file):
            existing_df = pd.read_excel(attendance_file)
            updated_df = pd.concat([existing_df, attendance_data], ignore_index=True)
        else:
            updated_df = attendance_data

        updated_df.to_excel(attendance_file, index=False)
        print("\n✅ تم تحديث ملف الحضور والغياب بنجاح.\n")

    except Exception as e:
        print(f"❌ خطأ أثناء تحديث ملف الحضور والغياب: {e}")



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--modelpath', type=str, default='weights/yolov8n-face.onnx', help="onnx filepath")
    parser.add_argument('--confThreshold', default=0.45, type=float, help='class confidence')
    parser.add_argument('--nmsThreshold', default=0.5, type=float, help='nms iou thresh')
    # We don't need the cam_url argument for a webcam
    args = parser.parse_args()

    # Initialize YOLOv8_face object detector
    YOLOv8_face_detector = YOLOv8_face(args.modelpath, conf_thres=args.confThreshold, iou_thres=args.nmsThreshold)

    # Initialize ArcFace recognizer
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    recognizer = iresnet_inference(model_name="r100", path="face_recognition/arcface/weights/arcface_r100.pth",
                                   device=device)

    # Load precomputed face features and names
    images_names, images_embs = read_features(feature_path="./datasets/face_features/feature")

    # Open video capture from webcam (index 0)
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Could not open video stream from the webcam.")
        exit()

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break

        # Detect faces in the frame
        boxes, scores, classids, kpts = YOLOv8_face_detector.detect(frame)

        if len(boxes) > 0:


            # Process each detected face
            for i, box in enumerate(boxes):
                x1, y1, w, h = map(int, box)
                x2, y2 = x1 + w, y1 + h
                face_img = frame[y1:y2, x1:x2]

                # Skip if face_img is invalid
                if face_img is None or face_img.size == 0:
                    print(f"Skipping invalid face image at index {i}")
                    continue

                # Align the face if landmarks are available and in the correct shape
                if len(kpts) > 0 and kpts[i].shape == (5, 2):
                    try:
                        face_img = norm_crop(frame, kpts[i])
                    except AssertionError as e:
                        print(f"Skipping alignment for face {i} due to incorrect landmark shape: {str(e)}")
                        continue

                # Recognize the face
                score, name = recognition(face_img, images_embs, images_names)
                if score is not None and score >= 0.25:
                    log_to_excel(name)


                # Display the recognition result
                caption = "UNKNOWN"
                if score is not None and score >= 0.25:
                    caption = f"{name}:{score:.2f}"
                cv2.putText(frame, caption, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)

                # Draw bounding box around face
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), thickness=2)

        # Display the frame
        cv2.imshow("Face Detection and Recognition", frame)

        # Press 'q' to exit
        if cv2.waitKey(1) & 0xFF == ord('m'):
            break

    finalize_attendance()
    cap.release()
    cv2.destroyAllWindows()
