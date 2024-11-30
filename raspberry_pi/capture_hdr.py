# import libraries
import picamera2
import cv2
import numpy as np
import json
import os
from typing import Dict
import tkinter as tk
from tkinter import Button
from PIL import Image, ImageTk

data_path = "./hdr_data"

default_exposure_times = [3000 * (2 ** (i-4)) for i in range(14)]

# Camera driver wrapper
class Camera:
    def __init__(self, exposure_times: list = []):
        # Initialize Picamera2 instance
        self.camera = picamera2.Picamera2()

        # Load default settings from camera
        if not exposure_times:
            # Set default exposure times if none are provided
            self.exposures = default_exposure_times
        else:
            self.exposures = exposure_times

        # Configure camera with base settings
        self.mode = self.camera.sensor_modes[1]
        sensor_config = {
            "output_size": self.mode['size'],
            "bit_depth": self.mode['bit_depth']
        }
        self.config = self.camera.create_still_configuration(sensor=sensor_config)
        self.camera.configure(self.config)
        self.camera.start()

    def set_config(self, config: dict):
        # Set configuration values from provided dictionary
        for key, value in config.items():
            if hasattr(self.camera.controls, key):
                setattr(self.camera.controls, key, value)

    def get_config(self) -> dict:
        # Get current configuration settings
        return self.camera.capture_metadata()

    def capture_image(self) -> np.ndarray:
        # Capture and return a single image as a NumPy array
        image = self.camera.capture_array()
        return image

    def warmup(self) -> dict:
        # Capture a few images to allow auto white balance and exposure to settle
        self.camera.stop()
        self.config['controls']['AwbEnable'] = True
        self.config['controls']['AwbMode'] = 0
        self.config['controls']['AeEnable'] = True
        self.config['controls']['AeExposureMode'] = 0
        self.config['controls']['AnalogueGain'] = 1.0
        self.camera.configure(self.config)
        self.camera.start()
        for _ in range(5):
            _ = self.capture_image()
        metadata = self.camera.capture_metadata()
        return {c: metadata[c] for c in ["ExposureTime", "AnalogueGain", "ColourGains"]}

    def capture_hdr(self) -> Dict[int, np.ndarray]:
        # Capture images at different exposure times and return them
        hdr_images = {}
        warmup_settings = self.warmup()
        print("Warmup settings after adjustment:", warmup_settings)
        self.set_config(warmup_settings)
        self.config['controls']['AeEnable'] = False
        self.config['controls']['AwbEnable'] = False

        for exposure_time in self.exposures:
            # Stop the camera, set the exposure time, and restart
            self.camera.stop()
            self.config['controls']['ExposureTime'] = exposure_time
            self.camera.configure(self.config)
            self.camera.start()
            image = self.capture_image()
            hdr_images[exposure_time] = image
        return hdr_images

# Save image data
def write_hdr(data_path: str, hdr_data: Dict[int, np.ndarray]):
    # Check if base directory exists, create if not
    if not os.path.exists(data_path):
        os.makedirs(data_path)

    # Determine the next scene number by counting existing subdirectories
    scene_number = len([name for name in os.listdir(data_path) if os.path.isdir(os.path.join(data_path, name))]) + 1
    scene_path = os.path.join(data_path, f"scene_{str(scene_number).zfill(4)}")
    os.makedirs(scene_path)

    # Save each image as a PNG file named by its exposure time
    for exposure_time, image in sorted(hdr_data.items()):
        image_path = os.path.join(scene_path, f"{exposure_time}.png")
        print(f"writing {image_path}")
        cv2.imwrite(image_path, cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

    return scene_path

# Save capture settings
def write_settings(data_path: str, camera_settings, exposure_times):
    settings_path = os.path.join(data_path, "camera_settings.json")
    settings_data = {
        "camera_settings": {key: value for key, value in camera_settings.items() if not key.startswith('_')},
        "exposure_times": exposure_times
    }
    with open(settings_path, 'w') as f:
        json.dump(settings_data, f, indent=4)

# GUI Application for Camera
class CameraApp:
    def __init__(self, root, camera):
        self.root = root
        self.camera = camera

        # Set up the GUI window
        self.gui_size = np.array([800, 400])
        self.root.title("Camera Viewfinder")
        # self.root.geometry(f"{self.gui_size[0]}x{self.gui_size[1]}")
        # self.root.state('normal')
        self.root.attributes("-zoomed", True)

        # Create a canvas to display the viewfinder
        self.canvas = tk.Label(self.root)
        self.canvas.pack()

        # Create a button to capture the HDR image
        self.capture_button = Button(self.root, text="Capture HDR Image", command=self.capture_hdr)
        self.capture_button.pack()

        # Start updating the viewfinder
        self.update_viewfinder()

    def update_viewfinder(self):
        # Capture an image and display it in the viewfinder
        frame = self.camera.capture_image()
        image_size = np.asarray(frame.shape[:2])
        resize_ratio = max(image_size/self.gui_size[::-1])
        frame = cv2.resize(frame, (image_size/resize_ratio).astype(int)[::-1]-50)
        image = Image.fromarray(frame)
        image_tk = ImageTk.PhotoImage(image=image)
        self.canvas.imgtk = image_tk
        self.canvas.configure(image=image_tk)
        
        # Schedule the next update
        self.root.after(50, self.update_viewfinder)

    def capture_hdr(self):
        # Capture HDR images
        hdr_images = self.camera.capture_hdr()
        scene_path = write_hdr(data_path, hdr_images)
        camera_settings = self.camera.get_config()
        write_settings(scene_path, camera_settings, default_exposure_times)
        print("HDR images and settings saved.")

if __name__ == '__main__':
    # Check if data_path exists. Create new directory if it does not.
    if not os.path.exists(data_path):
        os.makedirs(data_path)

    # Initialize Camera with exposure times
    camera = Camera()

    # Start the GUI application
    root = tk.Tk()
    app = CameraApp(root, camera)
    root.mainloop()

    # Release resources
    cv2.destroyAllWindows()
