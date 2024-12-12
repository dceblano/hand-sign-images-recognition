import matplotlib.pyplot as plt

def plot_the_metrics(history_finetune):
    history_finetune_dict = history_finetune.history

    # Extract accuracy and loss for training and validation
    fine_tune_train_acc = history_finetune_dict['accuracy']
    fine_tune_val_acc = history_finetune_dict['val_accuracy']
    fine_tune_train_loss = history_finetune_dict['loss']
    fine_tune_val_loss = history_finetune_dict['val_loss']

    # Define the number of epochs
    fine_tune_epochs = range(1, len(fine_tune_train_acc) + 1)

    # Set figure size and style
    plt.figure(figsize=(14, 6))

    # Plot Training and Validation Accuracy
    plt.subplot(1, 2, 1)
    plt.plot(fine_tune_epochs, fine_tune_train_acc, label='Training Accuracy (Fine-tuning)', color='blue')
    plt.plot(fine_tune_epochs, fine_tune_val_acc, label='Validation Accuracy (Fine-tuning)', color='orange')
    plt.title('Fine-Tuning: Training and Validation Accuracy', fontsize=14)
    plt.xlabel('Epochs', fontsize=12)
    plt.ylabel('Accuracy', fontsize=12)
    plt.legend()
    plt.grid(True)

    # Plot Training and Validation Loss
    plt.subplot(1, 2, 2)
    plt.plot(fine_tune_epochs, fine_tune_train_loss, label='Training Loss (Fine-tuning)', color='blue')
    plt.plot(fine_tune_epochs, fine_tune_val_loss, label='Validation Loss (Fine-tuning)', color='orange')
    plt.title('Fine-Tuning: Training and Validation Loss', fontsize=14)
    plt.xlabel('Epochs', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.legend()
    plt.grid(True)

    # Show plots
    plt.tight_layout()
    plt.show()