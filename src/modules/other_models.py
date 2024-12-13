from tensorflow.keras.applications import EfficientNetB0, VGG16
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam

import pickle

def save_history(history, filename):
    # Save the history dictionary using pickle
    with open(filename, 'wb') as f:
        pickle.dump(history.history, f)
    
# EfficientNetB0 setup
def setup_efficientnet_b0():
    base_model = EfficientNetB0(input_shape=(128, 128, 3), include_top=False, weights='imagenet')
    base_model.trainable = False
    model = Sequential([
        base_model,
        GlobalAveragePooling2D(),
        Dense(256, activation='relu'),
        Dense(24, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

# VGG16 setup
def setup_vgg16():
    base_model = VGG16(input_shape=(128, 128, 3), include_top=False, weights='imagenet')
    base_model.trainable = False
    model = Sequential([
        base_model,
        GlobalAveragePooling2D(),
        Dense(256, activation='relu'),
        Dense(24, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

# Callbacks setup for both models
def get_callbacks(model_name):
    early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    model_checkpoint = ModelCheckpoint(f'{model_name}_best_model.keras', monitor='val_loss', save_best_only=True, mode='min')
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-6)
    return [early_stopping, model_checkpoint, reduce_lr]

# Training and tuning function for both EfficientNetB0 and VGG16
def train_and_tune_other_models(training_set, test_set):
    # EfficientNetB0 setup and training
    efficientnet_model = setup_efficientnet_b0()
    efficientnet_callbacks = get_callbacks('efficientnet_b0')
    
    print("Training EfficientNetB0...")
    efficientnet_history = efficientnet_model.fit(
        training_set,
        validation_data=test_set,
        epochs=3, #10
        callbacks=efficientnet_callbacks
    )    

    # Example of saving history
    save_history(efficientnet_history, 'efficientnet_train_history.pkl')

    # Fine-tuning EfficientNetB0
    print("Fine-tuning EfficientNetB0...")
    for layer in efficientnet_model.layers[:-2]:  # Freeze all layers except the last 10
        layer.trainable = False
    
    efficientnet_model.compile(optimizer=Adam(learning_rate=0.0001), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    efficientnet_finetune_history = efficientnet_model.fit(
        training_set,
        validation_data=test_set,
        epochs=3, #5
        callbacks=efficientnet_callbacks
    )

    # Save the history to a pickle file
    save_history(efficientnet_finetune_history, 'efficientnet_finetune_history.pkl')
        
    # VGG16 setup and training
    vgg16_model = setup_vgg16()
    vgg16_callbacks = get_callbacks('vgg16')
    
    print("Training VGG16...")
    vgg16_history = vgg16_model.fit(
        training_set,
        validation_data=test_set,
        epochs=3, #10
        callbacks=vgg16_callbacks
    )

    # Save the history to a pickle file
    save_history(vgg16_history, 'vgg16_train_history.pkl')
      
    # Fine-tuning VGG16
    print("Fine-tuning VGG16...")
    for layer in vgg16_model.layers[:-1]:  # Freeze all layers except the last 5
        layer.trainable = False
    
    vgg16_model.compile(optimizer=Adam(learning_rate=0.0001), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    vgg16_finetune_history = vgg16_model.fit(
        training_set,
        validation_data=test_set,
        epochs=3, #5
        callbacks=vgg16_callbacks
    )

    # Save the history to a pickle file
    save_history(vgg16_finetune_history, 'vgg16_finetune_history.pkl')
    
    print("EfficientNetB0 Training History:", efficientnet_history.history)
    print("VGG16 Training History:", vgg16_history.history)


    return efficientnet_model, efficientnet_history, efficientnet_finetune_history, vgg16_model, vgg16_history, vgg16_finetune_history
