import numpy as np
import cv2
import json
import os
import random
import matplotlib.pyplot as plt

# Paths and settings
dataset_path = r'path/to/hdr_data'
labels_path = os.path.join(dataset_path, 'labels.json')

# Load labels from file
with open(labels_path, 'r') as f:
    labels = json.load(f)

# Function to select scene
selected_scene = input("Enter the name of the scene to use (or press Enter to select a random scene): ")
if not selected_scene:
    selected_scene = random.choice(labels)['scene']
    print(f"Selecting random scene: {selected_scene}")

# Find the scene label
scene_label = next((label for label in labels if label['scene'] == selected_scene), None)
if not scene_label:
    print(f"Scene '{selected_scene}' not found.")
    exit(1)

# Get the best exposure times from labels
best_exposure_times = [et for et in scene_label['best_exposure_times'] if et is not None]

# Get the image paths
scene_path = os.path.join(dataset_path, selected_scene)
images = [os.path.join(scene_path, img) for img in os.listdir(scene_path) if img.endswith('.png')]
images_dict = {
    float(os.path.basename(img_path).replace('.png', '')): img_path for img_path in images
}

# Select and load images
best_exposure_times.sort()
selected_images = [images_dict[exposure_time] for exposure_time in best_exposure_times if exposure_time in images_dict]
imgs = [cv2.imread(img_path) for img_path in selected_images]

# Plot the component images, histograms, and HDR result
num_images = len(imgs)
fig, axes = plt.subplots(3, max(num_images, 2), figsize=(5 * max(num_images, 2), 15))
fig.suptitle(f"HDR Merging for Scene: {selected_scene}", fontsize=16)

for i, img in enumerate(imgs):
    # Display component images
    axes[0, i].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    axes[0, i].set_title(f"Image {i+1} - Exposure Time: {best_exposure_times[i]}")
    axes[0, i].axis('off')

    # Load and display histograms from sorted labels
    hist = cv2.calcHist([cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)], [0], None, [256], [0, 256])
    axes[1, i].plot(hist, color='black')
    axes[1, i].set_xlim([0, 256])
    axes[1, i].set_title(f"Histogram for Image {i+1}")

# If there are at least two images, perform HDR merging
if len(imgs) >= 2:
    # Merge images using Mertens for HDR
    merge_mertens = cv2.createMergeMertens()
    ldr = merge_mertens.process(imgs)
    ldr = np.clip(ldr * 255, 0, 255).astype('uint8')

    # Display HDR result
    axes[2, 0].imshow(cv2.cvtColor(ldr, cv2.COLOR_BGR2RGB))
    axes[2, 0].set_title("Merged HDR Image")
    axes[2, 0].axis('off')

    # Display histogram for HDR result
    hdr_hist = cv2.calcHist([cv2.cvtColor(ldr, cv2.COLOR_BGR2GRAY)], [0], None, [256], [0, 256])
    axes[2, 1].plot(hdr_hist, color='black')
    axes[2, 1].set_xlim([0, 256])
    axes[2, 1].set_title("Histogram for Merged HDR Image")

# If there is only one image, display the image
elif len(imgs) == 1:
    img = imgs[0]
    axes[2, 0].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    axes[2, 0].set_title(f"Single Image - Exposure Time: {best_exposure_times[0]}")
    axes[2, 0].axis('off')

plt.tight_layout(rect=[0, 0, 1, 0.95], h_pad=2.0)
plt.show()
