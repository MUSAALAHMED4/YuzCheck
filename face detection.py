import cv2
import numpy as np
import os
import time
import argparse
import threading
import queue
from yolov8_face import YOLOv8_face

# Frame queue for threaded frame capture
frame_queue = queue.Queue(maxsize=1)

def parse_arguments():
    parser = argparse.ArgumentParser(description="YOLOv8 Face Detection within ROI")
    parser.add_argument('--modelpath', type=str, default='weights/yolov8n-face.onnx', help="Path to the ONNX model file")
    parser.add_argument('--confThreshold', default=0.45, type=float, help='Confidence threshold for detections')
    parser.add_argument('--nmsThreshold', default=0.5, type=float, help='NMS IoU threshold')
    parser.add_argument('--roi_x', type=int, default=500, help='Top-left x-coordinate of ROI')
    parser.add_argument('--roi_y', type=int, default=200, help='Top-left y-coordinate of ROI')
    parser.add_argument('--roi_w', type=int, default=2200, help='Width of ROI')
    parser.add_argument('--roi_h', type=int, default=1200, help='Height of ROI')
    parser.add_argument('--min_face_size', type=int, default=100, help='Minimum size (in pixels) for face width and height to be detected')
    parser.add_argument('--padding_size', type=int, default=10, help='Padding size (in pixels) around detected faces when saving images')
    parser.add_argument('--output_dir', type=str, default='detected_faces', help='Directory to save the cropped face images')
    return parser.parse_args()

def frame_reader(cap):
    """ Read frames from the camera and put the latest frame into the queue. """
    while True:
        ret, frame = cap.read()
        if not ret:
            continue
        # Keep only the most recent frame in the queue
        if not frame_queue.empty():
            frame_queue.get_nowait()
        frame_queue.put(frame)

if __name__ == '__main__':
    args = parse_arguments()

    # Initialize YOLOv8_face object detector
    YOLOv8_face_detector = YOLOv8_face(args.modelpath, conf_thres=args.confThreshold, iou_thres=args.nmsThreshold)

    # Create output directory if it doesn't exist
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    # Initialize a counter for saved images
    image_counter = 1

    # Open video capture from IP camera
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 0)  # Reduce buffer size to minimize latency

    if not cap.isOpened():
        print("Error: Could not open video stream from the IP camera.")
        exit()

    # Start frame reader thread
    threading.Thread(target=frame_reader, args=(cap,), daemon=True).start()

    while True:
        if not frame_queue.empty():
            # Get the latest frame from the queue
            frame = frame_queue.get()

            # Get the dimensions of the frame
            frame_height, frame_width = frame.shape[:2]

            # Define ROI coordinates
            roi_x = args.roi_x
            roi_y = args.roi_y
            roi_w = args.roi_w
            roi_h = args.roi_h

            # Ensure ROI is within frame boundaries
            roi_x = max(0, min(roi_x, frame_width - 1))
            roi_y = max(0, min(roi_y, frame_height - 1))
            roi_w = max(1, min(roi_w, frame_width - roi_x))
            roi_h = max(1, min(roi_h, frame_height - roi_y))

            # Extract ROI from the frame
            roi = frame[roi_y:roi_y + roi_h, roi_x:roi_x + roi_w]

            # Detect faces in the ROI
            boxes, scores, classids, kpts = YOLOv8_face_detector.detect(roi)

            # Filter out detections that are smaller than the minimum face size
            if args.min_face_size > 0 and len(boxes) > 0:
                valid_indices = []
                for i, box in enumerate(boxes):
                    _, _, w, h = box  # boxes are in [x, y, w, h] format
                    if w >= args.min_face_size and h >= args.min_face_size:
                        valid_indices.append(i)
                boxes = boxes[valid_indices]
                scores = scores[valid_indices]
                classids = classids[valid_indices]
                kpts = kpts[valid_indices]

            # Save detected faces with padding
            for box in boxes:
                x, y, w, h = box.astype(int)
                padding = args.padding_size

                # Calculate padded coordinates
                x_padded = max(x - padding, 0)
                y_padded = max(y - padding, 0)
                w_padded = w + 2 * padding
                h_padded = h + 2 * padding

                # Ensure the padded box does not exceed the ROI boundaries
                x_padded_end = min(x_padded + w_padded, roi_w)
                y_padded_end = min(y_padded + h_padded, roi_h)

                # Adjust width and height if padding goes beyond the ROI
                w_padded = x_padded_end - x_padded
                h_padded = y_padded_end - y_padded

                # Extract the face region with padding from the ROI
                face_roi = roi[y_padded:y_padded_end, x_padded:x_padded_end]

                # Construct a unique filename using a timestamp and counter
                timestamp = int(time.time() * 1000)  # Current time in milliseconds
                filename = f"pimage_{timestamp}_{image_counter}.jpg"
                filepath = os.path.join(args.output_dir, filename)

                # Save the face image
                cv2.imwrite(filepath, face_roi)
                print(f"Saved detected face to {filepath}")

                # Increment the image counter
                image_counter += 1

            # Draw detections on the original frame
            frame = YOLOv8_face_detector.draw_detections(frame, boxes, scores, kpts, roi=(roi_x, roi_y))

            # Draw ROI rectangle on the frame for visualization
            cv2.rectangle(
                frame,
                (roi_x, roi_y),
                (roi_x + roi_w, roi_y + roi_h),
                (255, 0, 0),
                thickness=2
            )

            # Dynamically resize the display window based on frame dimensions
            cv2.namedWindow("Face Detection within ROI", cv2.WINDOW_NORMAL)
            cv2.resizeWindow("Face Detection within ROI", frame_width, frame_height)

            # Display the frame
            cv2.imshow("Face Detection within ROI", frame)

            # Press 'q' to exit
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()
