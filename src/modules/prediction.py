import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report
from tensorflow.keras.preprocessing import image


def perform_prediction(model, training_set, test_set):
    
    # Initialize arrays for true labels and predictions
    true_labels = []
    predicted_labels = []

    # Loop through the test dataset in batches
    for images, labels in test_set:
        predictions = model.predict(images)  # Get predictions
        predicted_classes = np.argmax(predictions, axis=1)  # Convert probabilities to class indices
        true_labels.extend(labels)  # Store true labels
        predicted_labels.extend(predicted_classes)  # Store predicted labels

        # Break after processing all test data (important for generators)
        if len(true_labels) >= test_set.samples:
            break

    true_labels = np.array(true_labels)
    predicted_labels = np.array(predicted_labels)

    # -------------------------------
    # 1. Confusion Matrix
    # -------------------------------

    # Compute and display confusion matrix
    cm = confusion_matrix(true_labels, predicted_labels)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=list(training_set.class_indices.keys()))

    # Adjust figure size before plotting
    plt.figure(figsize=(10,10))  # Adjust the width and height as needed
    disp.plot(cmap='viridis', xticks_rotation='vertical', values_format='d')

    # Add title and show plot
    plt.title('Confusion Matrix', fontsize=20)
    plt.xlabel('Predicted Labels', fontsize=14)
    plt.ylabel('True Labels', fontsize=14)
    plt.show()

    # -------------------------------
    # 2. Classification Report
    # -------------------------------
    # Generate classification metrics
    print("Classification Report:\n")
    print(classification_report(true_labels, predicted_labels, target_names=list(training_set.class_indices.keys())))
    
    
    print('Display predictions on test images')
    plt.figure(figsize=(10, 10))
    test_images, test_labels = next(test_set)  # Get a batch of test data
    predictions = model.predict(test_images)

    for i in range(9):
        plt.subplot(3, 3, i + 1)
        plt.imshow(test_images[i].squeeze(), cmap='gray')
        predicted_label = np.argmax(predictions[i])
        true_label = test_labels[i]
        plt.title(f'Pred: {predicted_label}, True: {true_label}')
        plt.axis('off')
    plt.tight_layout()
    plt.show()
    
def predict_outside_image(image_path, model, class_indices):
    # Step 1: Load and preprocess the image
    test_image = image.load_img(image_path, target_size=(128, 128), color_mode='rgb')  # Resize to match model input
    test_image = image.img_to_array(test_image) / 255.0  # Normalize to [0, 1]
    test_image = np.expand_dims(test_image, axis=0)      # Add batch dimension

    # Step 2: Predict
    result = model.predict(test_image)
    predicted_class_index = np.argmax(result)

    # Step 3: Map prediction back to class label
    reverse_class_indices = {v: k for k, v in class_indices.items()}
    predicted_label = reverse_class_indices[predicted_class_index]

    # Step 4: Combine class labels and probabilities, then sort and select top 5
    probabilities = {reverse_class_indices[i]: prob for i, prob in enumerate(result[0])}
    top_5_predictions = sorted(probabilities.items(), key=lambda x: x[1], reverse=True)[:5]

    # Step 5: Print the predicted label and top 5 class probabilities
    print(f"Predicted Label: {predicted_label}")
    print("Top 5 Class Probabilities:")
    for label, prob in top_5_predictions:
        print(f"  {label}: {prob:.2f}")    