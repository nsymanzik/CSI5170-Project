import numpy as np
import json
import os

# Paths and settings
dataset_path = r'path/to/hdr_data'
histograms_path = os.path.join(dataset_path, 'histograms.json')

# Load histograms from file
with open(histograms_path, 'r') as f:
    image_histograms = json.load(f)

# Get list of scenes
dataset = []
labels = []

# Load dataset list of scenes
if os.path.exists(dataset_path):
    dataset = [os.path.join(dataset_path, scene) for scene in os.listdir(dataset_path) if os.path.isdir(os.path.join(dataset_path, scene))]

# Process each scene in the dataset
for scene in dataset:
    # Filter histograms for the current scene
    scene_histograms = [h for h in image_histograms if h['scene'] == os.path.basename(scene)]

    # Sort histograms by exposure time
    scene_histograms.sort(key=lambda x: float(x['image'].replace('.png', '')))

    # Initialize list of three exposure times
    best_exposure_times = [None, None, None]

    # Analyze each histogram to find the best low, mid, and high exposure times
    valid_images = []
    for hist_data in scene_histograms:
        hist = np.array(hist_data['histogram'])
        # Calculate percentage of very dark and very bright pixels
        dark_percentage = np.sum(hist[:25]) / np.sum(hist)
        bright_percentage = np.sum(hist[230:]) / np.sum(hist)

        # Filter out images that are too dark or too bright
        if dark_percentage < 0.9 and bright_percentage < 0.7:  # Tighter threshold on bright pixels
            # Calculate the weighted average intensity to represent brightness
            brightness_mean = np.sum(hist * np.arange(256)) / np.sum(hist)
            valid_images.append((hist_data['image'], brightness_mean))

    # Sort by brightness mean
    valid_images.sort(key=lambda x: x[1])

    # Assign low, mid, and high exposures from valid images
    if len(valid_images) >= 2:
        best_exposure_times[0] = float(valid_images[0][0].replace('.png', ''))  # Low exposure
        best_exposure_times[2] = float(valid_images[-1][0].replace('.png', ''))  # High exposure

        # Determine mid exposure time by selecting the median of the remaining candidates
        mid_candidates = valid_images[1:-1]
        if mid_candidates:
            mid_index = len(mid_candidates) // 2
            best_exposure_times[1] = float(mid_candidates[mid_index][0].replace('.png', ''))
    elif len(valid_images) == 1:
        # If only one valid image is available, set it as the mid exposure
        best_exposure_times[1] = float(valid_images[0][0].replace('.png', ''))

    # Append the labels for the scene
    labels.append({'scene': os.path.basename(scene), 'best_exposure_times': best_exposure_times})

# Save labels to file
labels_path = os.path.join(dataset_path, 'labels.json')
with open(labels_path, 'w') as f:
    json.dump(labels, f, indent=2)
