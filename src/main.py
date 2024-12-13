
import pickle
import numpy as np
import seaborn as sns
import tensorflow as tf
import matplotlib.pyplot as plt

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten, GlobalAveragePooling2D, BatchNormalization
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.callbacks import ReduceLROnPlateau
from tensorflow.keras.models import load_model

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report


from modules.data_preprocessing import perform_preprocessing
from modules.model_setup import setup_base_model, setup_model
from modules.model_training import train_the_model
from modules.model_tuning import tune_the_base_model, tune_the_model, perform_hyperparam_tuning
from modules.prediction import perform_prediction, predict_outside_image
from modules.plot_metrics import plot_the_metrics, plot_the_metrics2, plot_history_other_models, compare_fine_tune_histories
from modules.eda_processing import asl_eda_analysis
from modules.other_models import train_and_tune_other_models
from modules.load_and_print import load_and_print_metrics


def main():
    
    print('Step 0: Cleaning and EDA Analysis')
    asl_eda_analysis()

    print('\n Step 1: Loading and Preprocess Data')
    training_data, test_data = perform_preprocessing()
    
    print('\n Step 2: Load Pretrained Model (Feature Extractor)')
    base_model = setup_base_model()
    model = setup_model(base_model)
    
    print('\n Step 3: Hyper parameter tuning')
    best_hps = perform_hyperparam_tuning(training_data)
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
    
    model_checkpoint = train_the_model(model, training_data, test_data)
    
    print('\n Step 5: Model Tuning')
    base_model = tune_the_base_model(base_model)
    MobileNetV2_finetune_history = tune_the_model(model, training_data, test_data, model_checkpoint)
    efficientnet_model, efficientnet_train_history, efficientnet_finetune_history, vgg16_model, vgg16_train_history, vgg16_finetune_history  = train_and_tune_other_models(training_data, test_data)

    print('\n Step 6: Plot metrics')
    plot_the_metrics(MobileNetV2_finetune_history)
    plot_the_metrics2(MobileNetV2_finetune_history)

    plot_history_other_models(efficientnet_train_history, efficientnet_finetune_history, vgg16_train_history, vgg16_finetune_history)
    compare_fine_tune_histories(MobileNetV2_finetune_history, efficientnet_finetune_history, vgg16_finetune_history)

    '''
    # Load training and fine-tuning histories from predefined pickle files and print specific metrics.
    histories = load_and_print_metrics()

    # Use the loaded histories in other functions
    plot_history_other_models(
        histories["EfficientNet Training"], 
        histories["EfficientNet Fine-tuning"], 
        histories["VGG16 Training"], 
        histories["VGG16 Fine-tuning"]
    )

    compare_fine_tune_histories(
        histories["MobileNetV2 Fine-tuning"], 
        histories["EfficientNet Fine-tuning"], 
        histories["VGG16 Fine-tuning"]
    )
    '''

    print('\n Step 7: Predict on Entire Test Dataset')
    perform_prediction(model, training_data, test_data)

    # Predict using EfficientNetB0
    print("\nPredict using EfficientNetB0")
    perform_prediction(efficientnet_model, training_data, test_data)
    
    # Predict using VGG16
    print("\nPredict using VGG16")
    perform_prediction(vgg16_model, training_data, test_data)

    print('\n Step 8: Save the model')
    model.save('model/asl_recognition_model.h5')
    efficientnet_model.save('model/efficientnet_model.h5')
    vgg16_model.save('model/vgg16_model.h5')
    
    print('\n Step 9: Sample predict outside image')
    predict_outside_image('dataset/Prediction/A.jpg', model, training_data.class_indices)
    print("\n Predict Outside Image with EfficientNetB0")
    predict_outside_image('dataset/Prediction/A.jpg', efficientnet_model, training_data.class_indices)
    print("\n Predict Outside Image with VGG16")
    predict_outside_image('dataset/Prediction/A.jpg', vgg16_model, training_data.class_indices)


 
if __name__ == "__main__":
    main()