
import os
import re
import random
import hashlib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import plotly.express as px
from tensorflow.keras.preprocessing.image import load_img

# Function to Generate Image Hash
def get_image_hash(image_path):
    """Generate hash for an image to detect duplicates."""
    with open(image_path, "rb") as f:
        return hashlib.md5(f.read()).hexdigest()

def remove_files(file_list):
    for file in file_list:
        if os.path.exists(file):
            try:
                os.remove(file)
                print(f"Removed: {file}")
            except Exception as e:
                print(f"Error removing file {file}: {e}")
        else:
            print(f"File not found: {file}")

# Check for duplicates, corrupted images, and incorrect labels
def check_data_quality(image_dir, valid_classes):
    image_hashes_map = {}
    duplicates = []
    corrupted_images = []
    missing_labels = []

    for root, dirs, files in os.walk(image_dir):
        for file in files:
            image_path = os.path.join(root, file)
            try:
                img = Image.open(image_path)
                img.verify()  # Verify image integrity
                image_hash = get_image_hash(image_path)

                if image_hash in image_hashes_map:
                    duplicates.append((image_path, image_hashes_map[image_hash]))
                else:
                    image_hashes_map[image_hash] = image_path

                match = re.match(r'(\d+)_([A-Za-z]+)\.jpg$', file)
                if match:
                    label = match.group(2)
                    if label not in valid_classes:
                        missing_labels.append(file)
                else:
                    missing_labels.append(file)

            except (IOError, SyntaxError, OSError) as e:
                corrupted_images.append(image_path)
                print(f"Error processing {image_path}: dataset/Train{e}")

    return duplicates, corrupted_images, missing_labels


def asl_eda_analysis():
    # Define Valid Classes and Directories
    valid_classes = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y']
    train_dir = 'dataset/Train'
    print("Valid ASL Classes Defined:", valid_classes)

    # Handle duplicates, corrupted images, missing or incorrect labels
    print("\nChecking data quality...")
    duplicates_train, corrupted_train, missing_labels_train = check_data_quality(train_dir, valid_classes)

    if duplicates_train:
        print(f"\nFound {len(duplicates_train)} duplicate images in training set.")
        for dup in duplicates_train:
            print(f"Duplicate found: {dup}")
        remove_files([dup[0] for dup in duplicates_train])

    if corrupted_train:
        print(f"\nFound {len(corrupted_train)} corrupted images in training set.")
        for corrupted in corrupted_train:
            print(f"Corrupted image found: {corrupted}")
        remove_files(corrupted_train)

    if missing_labels_train:
        print(f"\nFound {len(missing_labels_train)} images with missing or incorrect labels in training set.")
        for missing in missing_labels_train:
            print(f"Missing or incorrect label: {missing}")
        remove_files(missing_labels_train)

    # Create a dictionary to count images per class
    print("\nCounting images per class...")
    class_counts = {cls: 0 for cls in valid_classes}
    for root, dirs, files in os.walk(train_dir):
        for file in files:
            match = re.match(r'(\d+)_([A-Za-z]+)\.jpg$', file)
            if match:
                label = match.group(2)
                if label in class_counts:
                    class_counts[label] += 1

    # Convert to DataFrame for visualization
    print("\nCreating class distribution DataFrame...")
    class_counts_df = pd.DataFrame.from_dict(class_counts, orient='index', columns=['Count']).reset_index()
    class_counts_df.columns = ['Class', 'Count']

    # Plot class distribution using Plotly
    print("\nPlotting class distribution...")
    fig = px.bar(class_counts_df, x='Class', y='Count', color='Class',
                 labels={'Class': 'ASL Gesture Classes', 'Count': 'Count'},
                 title='Class Distribution in Training Data', color_continuous_scale='Viridis')
    fig.update_xaxes(tickangle=45)
    fig.show()

    # Analyzing brightness distribution
    print("\nAnalyzing brightness distribution...")
    brightness_values = []
    for root, _, files in os.walk(train_dir):
        for file in files:
            img_path = os.path.join(root, file)
            try:
                with Image.open(img_path) as img:
                    grayscale = img.convert("L")  # Convert to grayscale
                    brightness_values.append(np.array(grayscale).mean())
            except Exception as e:
                print(f"Could not process {img_path}: {e}")

    print(f"Total brightness values: {len(brightness_values)}")

    # Plot brightness distribution using Plotly
    fig = px.histogram(brightness_values, nbins=30, title="Brightness Distribution",
                       labels={'value': 'Brightness Value'})
    fig.show()

    # Visualize random images from each class
    print("\nVisualizing random images from each class...")
    fig = plt.figure(figsize=(10, 10))
    for idx, cls in enumerate(valid_classes[:9]):
        class_dir = os.path.join(train_dir, cls)
        if os.path.exists(class_dir):
            img_name = random.choice(os.listdir(class_dir))
            img_path = os.path.join(class_dir, img_name)
            try:
                img = load_img(img_path, target_size=(128, 128))
                plt.subplot(3, 3, idx + 1)
                plt.imshow(img)
                plt.title(f'Class: {cls}')
                plt.axis('off')
            except Exception as e:
                print(f"Error loading image {img_path}: {e}")
    plt.suptitle('Sample Images from Each Class')
    plt.tight_layout()
    plt.show()

    print("\nASL EDA analysis complete.")
