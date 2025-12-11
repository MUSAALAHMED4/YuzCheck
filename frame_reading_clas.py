# frame_reading_clas.py

import cv2
import os
import time
from yolov8_face import YOLOv8_face


class FrameReadingClass:
    def __init__(self, model_path='weights/yolov8n-face.onnx', conf_thres=0.45, nms_thres=0.5,
                 min_face_size=50, padding_size=10, output_dir='detected_faces', frames_folder='frames'):
        self.model_path = model_path
        self.conf_thres = conf_thres
        self.nms_thres = nms_thres
        self.min_face_size = min_face_size
        self.padding_size = padding_size
        self.output_dir = output_dir
        self.frames_folder = frames_folder

        self.yolov8_face_detector = YOLOv8_face(
            self.model_path,
            conf_thres=self.conf_thres,
            iou_thres=self.nms_thres
        )

        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

        if not os.path.exists(self.frames_folder):
            raise FileNotFoundError(f"The folder '{self.frames_folder}' does not exist.")

    def process_frames(self):
        image_counter = 1

        while True:
            frame_files = [f for f in os.listdir(self.frames_folder) if os.path.isfile(os.path.join(self.frames_folder, f))]

            if len(frame_files) == 0:
                print("No more frames to process.")
                break

            for frame_file in frame_files:
                frame_path = os.path.join(self.frames_folder, frame_file)
                frame = cv2.imread(frame_path)
                if frame is None:
                    print(f"Error reading frame: {frame_file}")
                    continue

                boxes, scores, classids, kpts = self.yolov8_face_detector.detect(frame)

                if self.min_face_size > 0 and len(boxes) > 0:
                    valid_indices = [i for i, box in enumerate(boxes) if box[2] >= self.min_face_size and box[3] >= self.min_face_size]
                    boxes = boxes[valid_indices]
                    scores = scores[valid_indices]
                    classids = classids[valid_indices]
                    kpts = kpts[valid_indices]

                for box in boxes:
                    x, y, w, h = box.astype(int)
                    padding = self.padding_size
                    x_padded = max(x - padding, 0)
                    y_padded = max(y - padding, 0)
                    w_padded = w + 2 * padding
                    h_padded = h + 2 * padding
                    x_padded_end = min(x_padded + w_padded, frame.shape[1])
                    y_padded_end = min(y_padded + h_padded, frame.shape[0])
                    w_padded = x_padded_end - x_padded
                    h_padded = y_padded_end - y_padded

                    face_roi = frame[y_padded:y_padded_end, x_padded:x_padded_end]

                    timestamp = int(time.time() * 1000)
                    filename = f"pimage_{timestamp}_{image_counter}.jpg"
                    filepath = os.path.join(self.output_dir, filename)

                    cv2.imwrite(filepath, face_roi)
                    #print(f"Saved detected face to {filepath}")
                    image_counter += 1

                os.remove(frame_path)
                #print(f"Processed and deleted frame: {frame_file}")

        print("All frames processed and deleted.")
