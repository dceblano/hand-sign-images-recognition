from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.callbacks import ReduceLROnPlateau

def train_the_model(model, training_set, test_set):
    model_checkpoint = ModelCheckpoint(
        'model/best_model.keras',
        monitor='val_loss',
        save_best_only=True,
        mode='min'
    )
    
    '''
    Dynamically reduces the learning rate when the validation loss plateaus (does not improve for patience epochs).
    Helps the model converge better in later epochs by fine-tuning weights more cautiously.
    '''
    
    reduce_lr = ReduceLROnPlateau( # Dynamic adjustment of the learning rate during training can lead to better convergence.
        monitor='val_loss',
        factor=0.5,   # Reduce the learning rate by half
        patience=3,   # If no improvement for 3 epochs
        min_lr=1e-6   # Set a minimum learning rate
    )
    
    # early_stopping = EarlyStopping(
    #     monitor='val_loss', 
    #     patience=5, 
    #     restore_best_weights=True
    # )
    
    # Train the model
    history = model.fit(
        training_set,
        validation_data=test_set,
        epochs=1,
        callbacks=[reduce_lr, model_checkpoint]
    )
    
    return model_checkpoint