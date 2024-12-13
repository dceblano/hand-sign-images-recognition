from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import numpy as np

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
        'archive/Train',
        target_size=(128, 128),  # Resize to 128x128 pixels (Compatibility to MobileNetV2)
        batch_size=64,
        class_mode='sparse',     # Sparse categorical labels, returns labels as integers
        color_mode='rgb'   # RGB input because MobileNetV2 expects 3-channel RGB input
    )
    
    # Test Data Preprocessing
    test_datagen = ImageDataGenerator(rescale=1./255)
    test_set = test_datagen.flow_from_directory(
        'archive/Test',
        target_size=(128, 128),  # Match training size
        batch_size=64,
        class_mode='sparse',     # Sparse categorical labels
        color_mode='rgb'
    )

    # Display some augmented images
    x_batch, y_batch = next(training_set)  # Get a batch of training data
    plt.figure(figsize=(10, 10))
    for i in range(9):
        plt.subplot(3, 3, i + 1)
        plt.imshow(x_batch[i].squeeze(), cmap='gray')  # Use `cmap='gray'` for grayscale
        plt.title(f'Class: {y_batch[i]}')  # Sparse labels are integers
        plt.axis('off')
    plt.show()
    
    return train_datagen, training_set, test_set

# Extract data from the generator function
def extract_data_from_generator(generator):
    x_data = []
    y_data = []
    
    for i in range(len(generator)):
        x_batch, y_batch = generator[i]
        x_data.extend(x_batch)
        y_data.extend(y_batch)
    
    return np.array(x_data), np.array(y_data)