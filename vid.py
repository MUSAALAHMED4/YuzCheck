from tkinter import Label, Frame, Button, Tk, filedialog
import cv2
import threading
import os

os.environ['QT_QPA_PLATFORM'] = 'xcb'
import time
from yolov8_face import YOLOv8_face
from face_recognition_class import FaceRecognitionClass
import queue
from PIL import Image, ImageTk
import csv
from collections import deque
import pandas as pd
from datetime import datetime, timedelta


class AttendanceApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Video Attendance Application")
        self.root.geometry("800x600")

        # Video file path
        self.video_path = None

        # Excel logging components
        self.excel_file = 'attendance_log.xlsx'
        self.last_logged_times = {}

        # Create right frame for recognized faces and controls
        self.right_frame = Frame(self.root, width=400, height=600)
        self.right_frame.grid(row=0, column=1, padx=10, pady=10, sticky='n')

        # Add file selection button
        self.select_video_button = Button(
            self.right_frame,
            text="Select Video File",
            command=self.select_video_file,
            font=("Arial", 14),
            bg="blue",
            fg="white",
            width=15
        )
        self.select_video_button.pack(pady=10)

        # Add selected file label
        self.file_label = Label(self.right_frame, text="No file selected", font=("Arial", 12))
        self.file_label.pack(pady=5)

        # Labels for recognized faces
        self.recognized_faces_label = Label(self.right_frame, text="Last 3 Recognized Faces", font=("Arial", 14))
        self.recognized_faces_label.pack(pady=10)

        # Frame to hold the last three recognized face images
        self.faces_display_frame = Frame(self.right_frame)
        self.faces_display_frame.pack(pady=10)

        # Initialize deque to store last three faces
        self.last_three_faces = deque(maxlen=3)

        # Create three labels for displaying recognized faces
        self.face_image_labels = []
        for i in range(3):
            frame = Frame(self.faces_display_frame, width=200, height=200, bd=2, relief='sunken')
            frame.pack(side='left', padx=5)
            label = Label(frame, text=f"Face {i + 1}", font=("Arial", 12))
            label.pack(expand=True)
            self.face_image_labels.append(label)

        # Labels for the right panel
        self.recognized_name_label = Label(self.right_frame, text="Name: ", font=("Arial", 14))
        self.recognized_name_label.pack(pady=10)

        self.timestamp_label = Label(self.right_frame, text="Detection Time: ", font=("Arial", 14))
        self.timestamp_label.pack(pady=10)

        # Video progress information
        self.progress_label = Label(self.right_frame, text="Progress: 0%", font=("Arial", 12))
        self.progress_label.pack(pady=5)

        # Start and Stop buttons
        self.start_button = Button(
            self.right_frame,
            text="Start Recognition",
            command=self.start_recognition,
            font=("Arial", 14),
            bg="green",
            fg="white",
            width=15,
            state="disabled"  # Initially disabled until video is selected
        )
        self.start_button.pack(pady=20)

        self.stop_button = Button(
            self.right_frame,
            text="Stop Recognition",
            command=self.stop_recognition,
            font=("Arial", 14),
            bg="red",
            fg="white",
            state="disabled",
            width=15
        )
        self.stop_button.pack(pady=10)

        # Initialize video capture
        self.cap = None

        # Initialize face detection and recognition components
        self.face_detector = YOLOv8_face(
            'weights/yolov8n-face.onnx',
            0.45,
            0.3
        )

        self.face_recognizer = FaceRecognitionClass(
            model_name="r100",
            model_path="face_recognition/arcface/weights/arcface_r100.pth",
            feature_path="./datasets/face_features/feature",
            input_size=112,
            min_score=0.4
        )

        # Initialize CSV log
        self.log_file = 'attendance_log.csv'
        if not os.path.exists(self.log_file):
            with open(self.log_file, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['Name', 'Timestamp'])

        # Frame queue and events
        self.frame_queue = queue.Queue(maxsize=1)
        self.stop_event = threading.Event()

        # Initialize threads
        self.reader_thread = None
        self.processor_thread = None
        self.display_thread = None

        # GUI queue
        self.gui_queue = queue.Queue()

        # Video properties
        self.total_frames = 0
        self.current_frame = 0

        # Start GUI updater
        self.root.after(100, self.update_gui)

        self.video_label = Label(self.right_frame)
        self.video_label.pack(pady=10)

    def select_video_file(self):
        """Open file dialog to select video file"""
        filetypes = (
            ('Video files', '*.mp4 *.avi *.mkv'),
            ('All files', '*.*')
        )

        self.video_path = filedialog.askopenfilename(
            title='Select a video file',
            filetypes=filetypes
        )

        if self.video_path:
            self.file_label.config(text=f"Selected: {os.path.basename(self.video_path)}")
            self.start_button.config(state="normal")

            # Initialize video capture and get total frames
            temp_cap = cv2.VideoCapture(self.video_path)
            self.total_frames = int(temp_cap.get(cv2.CAP_PROP_FRAME_COUNT))
            temp_cap.release()

    def log_to_excel(self, name):
        """Log attendance to Excel file with once-per-minute restriction per person."""
        current_time = datetime.now()

        if name in self.last_logged_times:
            last_logged = self.last_logged_times[name]
            if current_time - last_logged < timedelta(minutes=1):
                return False

        self.last_logged_times[name] = current_time
        timestamp = current_time.strftime('%Y-%m-%d %H:%M:%S')
        data = {'Name': [name], 'Timestamp': [timestamp]}

        try:
            if os.path.exists(self.excel_file):
                existing_df = pd.read_excel(self.excel_file)
                new_df = pd.DataFrame(data)
                updated_df = pd.concat([existing_df, new_df], ignore_index=True)
            else:
                updated_df = pd.DataFrame(data)

            updated_df.to_excel(self.excel_file, index=False)
            print(f"\n{'=' * 50}\nNEW EXCEL LOG ENTRY at {timestamp}\nPerson: {name}\n{'=' * 50}\n")
            return True

        except Exception as e:
            print(f"Error logging to Excel: {e}")
            return False

    def start_recognition(self):
        """Start the face recognition and detection process."""
        if not self.video_path:
            print("Please select a video file first.")
            return

        self.cap = cv2.VideoCapture(self.video_path)
        if not self.cap.isOpened():
            print("Error: Could not open video file.")
            return

        self.current_frame = 0
        self.start_button.config(state="disabled")
        self.stop_button.config(state="normal")
        self.select_video_button.config(state="disabled")
        self.stop_event.clear()

        self.reader_thread = threading.Thread(target=self.frame_reader, daemon=True)
        self.reader_thread.start()

        self.processor_thread = threading.Thread(target=self.process_frames, daemon=True)
        self.processor_thread.start()

        self.display_thread = threading.Thread(target=self.show_video_feed, daemon=True)
        self.display_thread.start()

    def stop_recognition(self):
        """Stop the face recognition and detection process."""
        self.stop_event.set()
        self.start_button.config(state="normal")
        self.stop_button.config(state="disabled")
        self.select_video_button.config(state="normal")
        if self.cap:
            self.cap.release()

    def frame_reader(self):
        """Read frames from the video file and put them into the queue."""
        while not self.stop_event.is_set():
            try:
                if self.cap and self.cap.isOpened():
                    ret, frame = self.cap.read()
                    if not ret:
                        print("End of video reached")
                        self.stop_recognition()
                        break

                    self.current_frame += 1
                    progress = (self.current_frame / self.total_frames) * 100
                    self.progress_label.config(text=f"Progress: {progress:.1f}%")

                    if not self.frame_queue.empty():
                        try:
                            self.frame_queue.get_nowait()
                        except queue.Empty:
                            pass
                    self.frame_queue.put(frame)
                time.sleep(0.03)  # Control playback speed
            except Exception as e:
                print(f"Error in frame_reader: {e}")
                continue

    def process_frames(self):
        """Process frames for face detection and recognition."""
        while not self.stop_event.is_set():
            try:
                if not self.frame_queue.empty():
                    frame = self.frame_queue.get()

                    # Detect faces
                    boxes, scores, classids, kpts = self.face_detector.detect(frame)

                    # Filter small faces
                    min_face_size = 60
                    if min_face_size > 0 and len(boxes) > 0:
                        valid_indices = []
                        for i, box in enumerate(boxes):
                            _, _, w, h = box
                            if w >= min_face_size and h >= min_face_size:
                                valid_indices.append(i)
                        boxes = [boxes[i] for i in valid_indices]
                        scores = [scores[i] for i in valid_indices]
                        classids = [classids[i] for i in valid_indices]
                        kpts = [kpts[i] for i in valid_indices]

                    # Process detected faces
                    recognized_faces = []
                    for box in boxes:
                        x, y, w, h = box.astype(int)

                        # Extract face with padding
                        padding = 10
                        x_padded = max(x - padding, 0)
                        y_padded = max(y - padding, 0)
                        w_padded = min(w + 2 * padding, frame.shape[1] - x_padded)
                        h_padded = min(h + 2 * padding, frame.shape[0] - y_padded)

                        face_roi = frame[y_padded:y_padded + h_padded,
                                   x_padded:x_padded + w_padded]

                        # Recognize face
                        score, name = self.face_recognizer.recognize_face(face_roi)
                        if score is not None and score >= self.face_recognizer.min_score:
                            recognized_faces.append((name, face_roi.copy()))
                            self.log_recognition(name)
                            self.log_to_excel(name)

                    if recognized_faces:
                        self.gui_queue.put(recognized_faces)

                    # Draw detections
                    frame = self.face_detector.draw_detections(frame, boxes, scores, kpts)
                    self.gui_queue.put(frame)

                time.sleep(0.01)
            except Exception as e:
                print(f"Error in process_frames: {e}")
                continue

    from PIL import Image, ImageTk

    def show_video_feed(self):
        """Display the video feed using Tkinter instead of OpenCV"""
        while not self.stop_event.is_set():
            try:
                if not self.frame_queue.empty():
                    frame = self.frame_queue.get()

                    # تحويل الإطار من BGR إلى RGB لأن PIL يستخدم RGB
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    img = Image.fromarray(frame_rgb)
                    img = img.resize((400, 300))  # تعديل الحجم حسب الحاجة

                    # تحويل الصورة إلى تنسيق Tkinter
                    imgtk = ImageTk.PhotoImage(image=img)

                    # تحديث الملصق في واجهة Tkinter
                    self.video_label.config(image=imgtk)
                    self.video_label.image = imgtk  # تحديث الصورة للحفاظ عليها في الذاكرة

            except Exception as e:
                print(f"Error in show_video_feed: {e}")
                continue

        # إغلاق OpenCV عند الانتهاء
        cv2.destroyAllWindows()

    def log_recognition(self, name):
        """Log the recognized name and timestamp to the CSV file."""
        timestamp = time.strftime('%Y-%m-%d %H:%M:%S')
        with open(self.log_file, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([name, timestamp])

    from PIL import Image, ImageTk
    import cv2

    def update_gui_with_recognized_faces(self, faces):
        """Updated UI with newly discovered images and names."""
        if not isinstance(faces, list):
            return

        for entry in faces:
            if isinstance(entry, tuple) and len(entry) == 2:
                name, face_image = entry
                timestamp = time.strftime('%Y-%m-%d %H:%M:%S')

                if self.last_three_faces and self.last_three_faces[0][0] == name:
                    self.last_three_faces[0] = (name, timestamp, face_image)
                else:
                    self.last_three_faces.appendleft((name, timestamp, face_image))

        for idx, label in enumerate(self.face_image_labels):
            if idx < len(self.last_three_faces):
                name, timestamp, face_img = self.last_three_faces[idx]

                face_rgb = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
                img = Image.fromarray(face_rgb)
                img = img.resize((100, 100))

                imgtk = ImageTk.PhotoImage(image=img)

                label.config(image=imgtk, text=f"{name}\n{timestamp}", compound='top')
                label.image = imgtk
            else:
                label.config(text=f"Face {idx + 1}", image="")

    def update_gui(self):
        """Process GUI updates from the queue."""
        try:
            while not self.gui_queue.empty():
                recognized_faces = self.gui_queue.get_nowait()
                self.update_gui_with_recognized_faces(recognized_faces)
        except queue.Empty:
            pass
        finally:
            self.root.after(100, self.update_gui)

    def on_closing(self):
        """Handle the closing of the application."""
        self.stop_recognition()
        if self.cap:
            self.cap.release()
        cv2.destroyAllWindows()
        self.root.destroy()


if __name__ == "__main__":
    root = Tk()
    app = AttendanceApp(root)
    root.protocol("WM_DELETE_WINDOW", app.on_closing)
    root.mainloop()
