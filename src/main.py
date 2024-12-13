import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten, GlobalAveragePooling2D, BatchNormalization
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.callbacks import ReduceLROnPlateau
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report
import seaborn as sns

from modules.data_preprocessing import perform_preprocessing
from modules.model_setup import setup_base_model, setup_model
from modules.model_training import train_the_model
from modules.model_tuning import tune_the_base_model, tune_the_model, perform_hyperparam_tuning
from modules.prediction import perform_prediction, predict_outside_image
from modules.plot_metrics import plot_the_metrics

def main():
    print('\n Step 1: Loading and Preprocess Data')
    train_datagen, training_set, test_set = perform_preprocessing()
    
    print('\n Step 2: Load Pretrained Model (Feature Extractor)')
    base_model = setup_base_model()
    model = setup_model(base_model)
    
    print('\n Step 3: Hyper parameter tuning')
    best_hps = perform_hyperparam_tuning(training_set)
    print(best_hps)

    print('\n Step 4: Train the model')    
    # Recreate the base model and setup model again with the best hyperparameters
    base_model = setup_base_model()
    model = setup_model(base_model)
    
    model.compile(
        optimizer=Adam(learning_rate=best_hps.get('learning_rate')),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    model_checkpoint = train_the_model(model, training_set, test_set)
    
    print('\n Step 5: Model Tuning')
    base_model = tune_the_base_model(base_model)
    history_finetune = tune_the_model(model, training_set, test_set, model_checkpoint)
    
    print('\n Step 6: Predict on Entire Test Dataset')
    perform_prediction(model, training_set, test_set)
    
    print('\n Step 7: Plot metrics')
    plot_the_metrics(history_finetune)
    
    print('\n Step 8: Save the model')
    model.save('model/asl_recognition_model.h5')
    
    print('\n Step 9: Sample predict outside image')
    predict_outside_image('archive/Prediction/A.jpg', model, training_set.class_indices)

if __name__ == "__main__":
    main()