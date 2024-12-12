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
from modules.model_tuning import tune_the_base_model, tune_the_model
from modules.prediction import perform_prediction, predict_outside_image
from modules.plot_metrics import plot_the_metrics

def main():
    print('Step 1: Loading and Preprocess Data')
    train_datagen, training_set, test_set = perform_preprocessing()
    
    print('Step 2: Load Pretrained Model (Feature Extractor)')
    base_model = setup_base_model()
    model = setup_model(base_model)
    
    print('Step 3: Train the model')
    model_checkpoint = train_the_model(model, training_set, test_set)
    
    print('Step 4: Model Tuning')
    base_model = tune_the_base_model(base_model)
    history_finetune = tune_the_model(model, training_set, test_set, model_checkpoint)
    
    print('Step 5: Predict on Entire Test Dataset')
    perform_prediction(model, training_set, test_set)
    
    print('Step 6: Plot metrics')
    plot_the_metrics
    
    print('Step 7: Save the model')
    model.save('model/asl_recognition_model.h5')
    
    print('Step 8: Sample predict outside image')
    predict_outside_image('archive/Prediction/A.jpg', model, training_set.class_indices)

if __name__ == "__main__":
    main()