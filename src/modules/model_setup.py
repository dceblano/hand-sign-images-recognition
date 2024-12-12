from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten, GlobalAveragePooling2D, BatchNormalization
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.callbacks import ReduceLROnPlateau

def setup_base_model():
    base_model = MobileNetV2(
        input_shape=(128, 128, 3),  # MobileNetV2 expects 3-channel RGB input
        include_top=False,          # Exclude the classification head, Excludes dense layer, acts as feature extractor
        weights='imagenet'          # Use pretrained weights
    )

    # Freeze the base model's layers to prevent training
    base_model.trainable = False
    return base_model

def setup_model(base_model):
    # Add custom layers on top of the pretrained base model
    model = Sequential([

        # Use the pretrained base model as a fixed feature extractor
        base_model,  # Pretrained MobileNetV2 without the classification head
        GlobalAveragePooling2D(),  # Replace Flatten with GAP to reduce overfitting

        # Custom dense layers for classification
        Dense(256, activation='relu'),
        BatchNormalization(), # Normalizes the inputs to the dense layer, speeding up training and stabilizing the learning process.
        Dropout(0.5),  # Dropout to prevent overfitting
        Dense(128, activation='relu'),
        Dropout(0.5),  # Dropout to prevent overfitting
        Dense(24, activation='softmax')  # 24 classes for ASL
    ])
    
    model.compile(
        optimizer=Adam(learning_rate=0.00001),  # Lower learning rate for stability
        loss='sparse_categorical_crossentropy', # Labels are integers
        metrics=['accuracy']
    )
    return model