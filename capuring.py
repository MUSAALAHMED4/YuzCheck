import os
import cv2
import time

def capture_images(name, save_dir, num_images=30, delay=0.25):
    """
    Capture images from the webcam.

    Args:
        name (str): The name of the person.
        save_dir (str): The directory where the images will be saved.
        num_images (int): Number of images to capture.
        delay (float): Delay between captures in seconds.
    """
    cap = cv2.VideoCapture(0)
    person_folder = os.path.join(save_dir, name)
    os.makedirs(person_folder, exist_ok=True)

    print(f"Press 'c' to start capturing {num_images} images for {name}.")
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to capture image")
            break

        cv2.imshow('Preview', frame)

        # Wait for the 'c' key to start capturing images
        if cv2.waitKey(1) & 0xFF == ord('c'):
            print(f"Starting capture for {name}.")
            for i in range(num_images):
                ret, frame = cap.read()
                if not ret:
                    print("Failed to capture image")
                    break
                cv2.imwrite(os.path.join(person_folder, f"{i}.jpg"), frame)
                time.sleep(delay)
                cv2.imshow('Capture', frame)

                # Check if 'q' is pressed to quit early
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            break

    cap.release()
    cv2.destroyAllWindows()
    print(f"Captured images saved in {person_folder}")

if __name__ == "__main__":
    person_name = input("Enter the name of the new person: ")
    save_directory = "./datasets/new_persons"
    capture_images(person_name, save_directory)
