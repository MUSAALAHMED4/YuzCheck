import cv2
import os
import time

def save_frames_from_roi(cam_url, roi_x, roi_y, roi_w, roi_h, output_dir, fps=20):
    # Create the output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Open video capture from IP camera
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Could not open video stream from the IP camera.")
        return

    # Calculate the time interval between frames based on the FPS
    frame_interval = 1 / fps
    last_saved_time = time.time()

    frame_counter = 1

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break

        current_time = time.time()
        if current_time - last_saved_time >= frame_interval:
            frame_height, frame_width = frame.shape[:2]

            # Define and ensure the ROI is within frame boundaries
            roi_x = max(0, min(roi_x, frame_width - 1))
            roi_y = max(0, min(roi_y, frame_height - 1))
            roi_w = max(1, min(roi_w, frame_width - roi_x))
            roi_h = max(1, min(roi_h, frame_height - roi_y))

            # Extract ROI from the frame
            roi = frame[roi_y:roi_y + roi_h, roi_x:roi_x + roi_w]

            # Save the ROI as an image
            filename = f"frame_{frame_counter}.jpg"
            filepath = os.path.join(output_dir, filename)
            cv2.imwrite(filepath, roi)

            print(f"Saved ROI frame to {filepath}")

            # Update last saved time and increment frame counter
            last_saved_time = current_time
            frame_counter += 1

        # Display the frame with ROI for visualization
        cv2.rectangle(frame, (roi_x, roi_y), (roi_x + roi_w, roi_y + roi_h), (255, 0, 0), 2)
        cv2.imshow("ROI Frame", frame)

        # Press 'q' to exit the loop
        if cv2.waitKey(1) & 0xFF == ord('m'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    # Define camera URL and ROI parameters
    cam_url = 'rtsp://admin:Bb@363926@192.168.1.164:554/stream1'
    roi_x = 200
    roi_y = 280
    roi_w = 1500
    roi_h = 550

    # Define the output directory for saving frames
    output_dir = 'frames'

    # Call the function to save frames from the ROI at 20 FPS
    save_frames_from_roi(cam_url, roi_x, roi_y, roi_w, roi_h, output_dir, fps=20)
