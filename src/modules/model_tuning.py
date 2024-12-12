from tensorflow.keras.optimizers import Adam

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