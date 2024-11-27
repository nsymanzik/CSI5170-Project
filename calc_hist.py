import numpy as np
import cv2
import json
import os

# Paths and settings
dataset_path = r'C:\Users\anony\Desktop\CSI5170\Project\hdr_data'
histograms_path = os.path.join(dataset_path, 'histograms.json')

# Get list of scenes
dataset = []
image_histograms = []

# Load dataset list of scenes
if os.path.exists(dataset_path):
    dataset = [os.path.join(dataset_path, scene) for scene in os.listdir(dataset_path) if os.path.isdir(os.path.join(dataset_path, scene))]

# Process each scene in the dataset
for scene in dataset:
    # Get list of images for the scene
    images = [os.path.join(scene, img) for img in os.listdir(scene) if img.endswith('.png')]

    # Analyze each image to calculate its histogram
    for img_path in images:
        print(f"Processing {img_path}")

        # Load the image
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            print(f"Failed to load image: {img_path}")
            continue

        # Calculate the normalized histogram
        hist = cv2.calcHist([img], [0], None, [256], [0, 256])
        hist = hist / np.sum(hist)

        # Store histogram information
        image_histograms.append({
            'scene': os.path.basename(scene),
            'image': os.path.basename(img_path),
            'histogram': hist.tolist()
        })

# Save histograms to file
with open(histograms_path, 'w') as f:
    json.dump(image_histograms, f, indent=2)

print(f"Histograms saved to {histograms_path}")
