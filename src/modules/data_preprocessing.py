from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import pickle
import seaborn as sns
import plotly.express as px

def perform_preprocessing():
    # Data Augmentation
    train_datagen = ImageDataGenerator(
        rescale=1./255, # Normalizes pixel values from the range [0, 255] to [0, 1] by dividing every pixel by 255.
        rotation_range=30,
        width_shift_range=0.2,
        height_shift_range=0.2,
        brightness_range=[0.8, 1.2],
        horizontal_flip=True,
        zoom_range=0.2,
    )

    training_set = train_datagen.flow_from_directory(
        'dataset/Train',
        target_size=(128, 128),  # Resize to 128x128 pixels (Compatibility to MobileNetV2)
        batch_size=64,
        class_mode='sparse',     # Sparse categorical labels, returns labels as integers
        color_mode='rgb'   # RGB input because MobileNetV2 expects 3-channel RGB input
    )
    
    # Test Data Preprocessing
    test_datagen = ImageDataGenerator(rescale=1./255)
    test_set = test_datagen.flow_from_directory(
        'dataset/Test',
        target_size=(128, 128),  # Match training size
        batch_size=64,
        class_mode='sparse',     # Sparse categorical labels
        color_mode='rgb'
    )
    '''
    # Load the test data and split it into validation and test sets
    test_set = test_datagen.flow_from_directory(
        'dataset/Test',
        target_size=(128, 128),  # Match training size
        batch_size=64,
        class_mode='sparse',     # Sparse categorical labels
        color_mode='rgb',
        subset='training'        # This will use the first 50% for training (validation here)
    )

    val_set = test_datagen.flow_from_directory(
        'dataset/Test',
        target_size=(128, 128),  # Match training size
        batch_size=64,
        class_mode='sparse',     # Sparse categorical labels
        color_mode='rgb',
        subset='validation'      # This will use the second 50% for validation
    )
    '''
    # Display some augmented images
    x_batch, y_batch = next(training_set)  # Get a batch of training data
    plt.figure(figsize=(10, 10))
    for i in range(9):
        plt.subplot(3, 3, i + 1)
        plt.imshow(x_batch[i].squeeze(), cmap='gray')  # Use `cmap='gray'` for grayscale
        plt.title(f'Class: {y_batch[i]}')  # Sparse labels are integers
        plt.axis('off')
    plt.show()
    
    class_indices = training_set.class_indices  # Get the class label-to-index mapping
    with open('model/class_indices.pkl', 'wb') as f:
        pickle.dump(class_indices, f)  # Save the mapping to a file
        
    reverse_class_indices = get_reverse_class_indices(training_set)  # Get the reverse mapping
    
    plot_class_distribution(reverse_class_indices, training_set)
    
    return training_set, test_set

def get_reverse_class_indices(training_set):
    return {v: k for k, v in training_set.class_indices.items()}

def plot_class_distribution(reverse_class_indices, training_set):
    # Create a count plot
    class_names = [reverse_class_indices[label] for label in training_set.classes]
    df = pd.DataFrame({'Class': class_names})
    # Create the count plot
    fig = px.histogram(df, x="Class", category_orders={"Class": sorted(reverse_class_indices.values())},
                    title="Class Distribution in Training Data", labels={"Class": "Classes"})

    # Customize axes labels and title
    fig.update_layout(
        xaxis_title="Classes",
        yaxis_title="Number of Images",
        xaxis_tickangle=90  # Rotate class names for better visibility
    )

    fig.show()

# Extract data from the generator function
def extract_data_from_generator(generator):
    x_data = []
    y_data = []
    
    for i in range(len(generator)):
        x_batch, y_batch = generator[i]
        x_data.extend(x_batch)
        y_data.extend(y_batch)
    
    return np.array(x_data), np.array(y_data)