import os
import cv2
import numpy as np
from tkinter import Tk, Label, Button, Entry, messagebox, Frame
from PIL import Image, ImageTk

class ImageProcessor:
    def __init__(self, input_dir, save_dir):
        self.input_dir = input_dir
        self.save_dir = save_dir
        self.image_files = [
            f for f in os.listdir(input_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif'))
        ]
        self.current_index = 0

        # Setup Tkinter Window
        self.root = Tk()
        self.root.title("Image Processor")

        # Frame for Image and Controls
        self.frame = Frame(self.root)
        self.frame.pack(padx=10, pady=10)

        # Label to Display Image
        self.image_label = Label(self.frame)
        self.image_label.grid(row=0, column=0, columnspan=3)

        # Entry for Person Name
        self.name_entry = Entry(self.frame, width=30)
        self.name_entry.grid(row=1, column=0, padx=10, pady=5)

        # Add Person Button
        self.add_button = Button(self.frame, text="Add Person", command=self.save_image)
        self.add_button.grid(row=1, column=1, padx=10, pady=5)

        # Skip Button
        self.skip_button = Button(self.frame, text="Skip", command=self.skip_image)
        self.skip_button.grid(row=1, column=2, padx=10, pady=5)

        self.load_image()
        self.root.mainloop()

    def load_image(self):
        """Load and display the current image and auto-fill the name entry."""
        if self.current_index >= len(self.image_files):
            messagebox.showinfo("Info", "All images processed.")
            self.root.quit()
            return

        filename = self.image_files[self.current_index]
        file_path = os.path.join(self.input_dir, filename)

        # Read and convert the image to display with Tkinter
        image = cv2.imdecode(np.fromfile(file_path, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
        if image is None:
            messagebox.showwarning("Warning", f"Failed to load {filename}. Skipping...")
            self.skip_image()
            return

        # Display the image
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(image)
        image = image.resize((500, 400))  # Resize for display purposes
        image_tk = ImageTk.PhotoImage(image)

        self.image_label.configure(image=image_tk)
        self.image_label.image = image_tk  # Keep a reference to avoid garbage collection

        # Auto-fill the entry field with the image's name (without extension)
        image_name = os.path.splitext(filename)[0]
        self.name_entry.delete(0, 'end')
        self.name_entry.insert(0, image_name)

    def skip_image(self):
        """Skip the current image."""
        self.current_index += 1
        self.load_image()

    def save_image(self):
        """Save the current image with the entered name."""
        name = self.name_entry.get().strip()
        if not name:
            messagebox.showwarning("Warning", "Please enter a name.")
            return

        person_folder = os.path.join(self.save_dir, name)
        os.makedirs(person_folder, exist_ok=True)

        filename = self.image_files[self.current_index]
        file_path = os.path.join(self.input_dir, filename)

        # Save the image as '0.png' in the person's folder
        new_filename = os.path.join(person_folder, "0.png")
        image = cv2.imdecode(np.fromfile(file_path, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
        cv2.imwrite(new_filename, image, [cv2.IMWRITE_PNG_COMPRESSION, 9])

        messagebox.showinfo("Info", f"Image saved as {new_filename}.")
        self.current_index += 1
        self.load_image()

# Example usage
input_directory = r"/Users/user/EDU2/Yazılım Projesi Geliştirme/facedetectionv2/images"
output_directory = r"/Users/user/EDU2/Yazılım Projesi Geliştirme/facedetectionv2/datasets/new_persons"


ImageProcessor(input_directory, output_directory)
