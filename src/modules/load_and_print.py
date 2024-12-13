import pickle

def load_and_print_metrics():
    """
    Load training and fine-tuning histories from predefined pickle files and print specific metrics.
    """
    history_files = {
        "MobileNetV2 Fine-tuning": "MobileNetV2_finetune_history.pkl",
        "EfficientNet Training": "efficientnet_train_history.pkl",
        "EfficientNet Fine-tuning": "efficientnet_finetune_history.pkl",
        "VGG16 Training": "vgg16_train_history.pkl",
        "VGG16 Fine-tuning": "vgg16_finetune_history.pkl"
    }
    
    histories = {}
    for model_name, file_path in history_files.items():
        try:
            with open(file_path, 'rb') as f:
                history = pickle.load(f)
                histories[model_name] = history
                # Print specific metrics
                print(f"{model_name} Training Accuracy:", history['accuracy'])
                print(f"{model_name} Validation Accuracy:", history['val_accuracy'])
                print(f"{model_name} Training Loss:", history['loss'])
                print(f"{model_name} Validation Loss:", history['val_loss'])
                print("-" * 50)
        except Exception as e:
            print(f"Error loading {model_name} history from {file_path}: {e}")
    
    return histories