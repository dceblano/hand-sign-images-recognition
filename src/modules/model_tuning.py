import keras_tuner as kt
from scikeras.wrappers import KerasClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers
from keras.applications import MobileNetV2
from keras.layers import GlobalAveragePooling2D, Dense, Dropout, BatchNormalization
from keras.optimizers import Adam
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from modules.model_setup import setup_base_model
from modules.data_preprocessing import extract_data_from_generator

# Define a function to create the model
def create_model(hp):
    base_model = setup_base_model()
    
    model = Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dense(256, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(hp.Choice('dropout_rate', [0.0, 0.2, 0.5])),
        layers.Dense(128, activation='relu'),
        layers.Dropout(hp.Choice('dropout_rate', [0.0, 0.2, 0.5])),
        layers.Dense(24, activation='softmax')
    ])
    
    model.compile(
        optimizer=Adam(learning_rate=hp.Choice('learning_rate', [0.01, 0.001, 0.0001])),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    return model

# Function to perform hyperparameter tuning using Hyperband
def perform_hyperparam_tuning(training_set):
    # Extract data from the generator
    x_train, y_train = extract_data_from_generator(training_set)
    
    tuner = kt.Hyperband(
        create_model,
        objective='val_accuracy',
        max_epochs=2,
        factor=3,
        directory='hyperband_dir',
        project_name='hyperband_tuning'
    )
    
    tuner.search(x_train, y_train, epochs=2, validation_split=0.2)
    
    # Get the optimal hyperparameters
    best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
    
    print(f"Best Hyperparameters: Learning Rate: {best_hps.get('learning_rate')}, Dropout Rate: {best_hps.get('dropout_rate')}")
    return best_hps

def tune_the_base_model(base_model):
    # Unfreeze the base model for fine-tuning
    base_model.trainable = True # Unfreeze to slightly adjust the pretrained weights

    # Unfreezing deeper layers since they capture high-level features relevant
    for layer in base_model.layers[:-50]:  # Freeze all layers except the last 50
        layer.trainable = False
        
def tune_the_model(model, training_set, test_set, model_checkpoint):
    model.compile(
        optimizer=Adam(learning_rate=0.0001),  # incerease LR since  frozen layers have already stabilized.
        loss='sparse_categorical_crossentropy', 
        metrics=['accuracy']
    )
    
    history_finetune = model.fit(
        training_set,
        validation_data=test_set,
        epochs=1,  # Train for a few more epochs
        callbacks=[model_checkpoint]  # Apply early stopping
    )
    return history_finetune