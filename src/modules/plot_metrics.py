import plotly.graph_objs as go
import plotly.subplots as sp
import matplotlib.pyplot as plt
from plotly.subplots import make_subplots

def plot_the_metrics(history_finetune):
    history_finetune_dict = history_finetune.history

    # Extract accuracy and loss for training and validation
    fine_tune_train_acc = history_finetune_dict['accuracy']
    fine_tune_val_acc = history_finetune_dict['val_accuracy']
    fine_tune_train_loss = history_finetune_dict['loss']
    fine_tune_val_loss = history_finetune_dict['val_loss']

    # Define the number of epochs
    fine_tune_epochs = list(range(1, len(fine_tune_train_acc) + 1))

    # Create subplots
    fig = sp.make_subplots(rows=1, cols=2, subplot_titles=('Fine-Tuning: Training and Validation Accuracy', 'Fine-Tuning: Training and Validation Loss'))

    # Add traces for accuracy
    fig.add_trace(go.Scatter(x=fine_tune_epochs, y=fine_tune_train_acc, mode='lines', name='Training Accuracy (Fine-tuning)', line=dict(color='blue')), row=1, col=1)
    fig.add_trace(go.Scatter(x=fine_tune_epochs, y=fine_tune_val_acc, mode='lines', name='Validation Accuracy (Fine-tuning)', line=dict(color='orange')), row=1, col=1)

    # Add traces for loss
    fig.add_trace(go.Scatter(x=fine_tune_epochs, y=fine_tune_train_loss, mode='lines', name='Training Loss (Fine-tuning)', line=dict(color='blue')), row=1, col=2)
    fig.add_trace(go.Scatter(x=fine_tune_epochs, y=fine_tune_val_loss, mode='lines', name='Validation Loss (Fine-tuning)', line=dict(color='orange')), row=1, col=2)

    # Update layout
    fig.update_layout(
        title='Fine-Tuning: Training and Validation Metrics',
        xaxis_title='Epochs',
        yaxis_title='Value',
        height=600,
        width=1200,
        showlegend=True
    )

    # Update xaxis and yaxis titles for each subplot
    fig.update_xaxes(title_text='Epochs', row=1, col=1)
    fig.update_yaxes(title_text='Accuracy', row=1, col=1)
    fig.update_xaxes(title_text='Epochs', row=1, col=2)
    fig.update_yaxes(title_text='Loss', row=1, col=2)

    # Show plot
    fig.show()

# Example usage
# Assuming 'history_finetune' is the history object returned from model training
# plot_the_metrics(history_finetune)


def plot_the_metrics2(history_finetune):
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


# Function to plot history using Plotly
def plot_history(history, title):
    # Check if history is a dictionary (loaded from pickle)
    if isinstance(history, dict):
        history_dict = history  # If it's a dictionary, use it directly
    else:
        history_dict = history.history  # If it's a Keras History object, access .history

    # Create subplots: 1 row, 2 columns
    fig = make_subplots(
        rows=1, cols=2, 
        subplot_titles=['Accuracy (Train vs Validation)', 'Loss (Train vs Validation)']
    )
    
    # Add trace for Training Accuracy
    fig.add_trace(go.Scatter(
        x=list(range(len(history_dict['accuracy']))),
        y=history_dict['accuracy'],
        mode='lines',
        name='Train Accuracy',
        line=dict(color='blue')),
        row=1, col=1
    )
    
    # Add trace for Validation Accuracy
    fig.add_trace(go.Scatter(
        x=list(range(len(history_dict['val_accuracy']))),
        y=history_dict['val_accuracy'],
        mode='lines',
        name='Validation Accuracy',
        line=dict(color='orange')),
        row=1, col=1
    )
    
    # Add trace for Training Loss
    fig.add_trace(go.Scatter(
        x=list(range(len(history_dict['loss']))),
        y=history_dict['loss'],
        mode='lines',
        name='Train Loss',
        line=dict(color='blue')),
        row=1, col=2
    )
    
    # Add trace for Validation Loss
    fig.add_trace(go.Scatter(
        x=list(range(len(history_dict['val_loss']))),
        y=history_dict['val_loss'],
        mode='lines',
        name='Validation Loss',
        line=dict(color='orange')),
        row=1, col=2
    )
    
    # Update layout
    fig.update_layout(
        title=title,
        xaxis_title='Epochs',
        yaxis_title='Metrics',
        height=500,
        width=1000,
        legend_title='Metrics',
        template='plotly_dark'  # Optional, you can change the template
    )
    
    # Show the figure
    fig.show()


# Function to plot the histories of both models (training and fine-tuning)
def plot_history_other_models(efficientnet_train_history, efficientnet_finetune_history, vgg16_train_history, vgg16_finetune_history):
    # Plot EfficientNetB0 training and fine-tuning history
    plot_history(efficientnet_train_history, "EfficientNetB0 Training")
    plot_history(efficientnet_finetune_history, "EfficientNetB0 Fine-Tuning")

    # Plot VGG16 training and fine-tuning history
    plot_history(vgg16_train_history, "VGG16 Training")
    plot_history(vgg16_finetune_history, "VGG16 Fine-Tuning")


def compare_fine_tune_histories(mobilenet_history, efficientnet_history, vgg16_history):
    # Helper function to extract history dictionary
    def get_history_dict(history):
        return history.history if not isinstance(history, dict) else history

    # Extract histories
    mobilenet = get_history_dict(mobilenet_history)
    efficientnet = get_history_dict(efficientnet_history)
    vgg16 = get_history_dict(vgg16_history)

    # Create subplots: 1 row, 2 columns (Accuracy and Loss)
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=["Fine-Tuning Accuracy Comparison", "Fine-Tuning Loss Comparison"]
    )

    # Add Accuracy Traces
    fig.add_trace(go.Scatter(
        x=list(range(len(mobilenet['accuracy']))),
        y=mobilenet['accuracy'],
        mode='lines',
        name='MobileNetV2 Train Accuracy',
        line=dict(color='blue')),
        row=1, col=1
    )
    fig.add_trace(go.Scatter(
        x=list(range(len(mobilenet['val_accuracy']))),
        y=mobilenet['val_accuracy'],
        mode='lines',
        name='MobileNetV2 Validation Accuracy',
        line=dict(color='blue', dash='dash')),
        row=1, col=1
    )
    fig.add_trace(go.Scatter(
        x=list(range(len(efficientnet['accuracy']))),
        y=efficientnet['accuracy'],
        mode='lines',
        name='EfficientNet Train Accuracy',
        line=dict(color='green')),
        row=1, col=1
    )
    fig.add_trace(go.Scatter(
        x=list(range(len(efficientnet['val_accuracy']))),
        y=efficientnet['val_accuracy'],
        mode='lines',
        name='EfficientNet Validation Accuracy',
        line=dict(color='green', dash='dash')),
        row=1, col=1
    )
    fig.add_trace(go.Scatter(
        x=list(range(len(vgg16['accuracy']))),
        y=vgg16['accuracy'],
        mode='lines',
        name='VGG16 Train Accuracy',
        line=dict(color='orange')),
        row=1, col=1
    )
    fig.add_trace(go.Scatter(
        x=list(range(len(vgg16['val_accuracy']))),
        y=vgg16['val_accuracy'],
        mode='lines',
        name='VGG16 Validation Accuracy',
        line=dict(color='orange', dash='dash')),
        row=1, col=1
    )

    # Add Loss Traces
    fig.add_trace(go.Scatter(
        x=list(range(len(mobilenet['loss']))),
        y=mobilenet['loss'],
        mode='lines',
        name='MobileNetV2 Train Loss',
        line=dict(color='blue')),
        row=1, col=2
    )
    fig.add_trace(go.Scatter(
        x=list(range(len(mobilenet['val_loss']))),
        y=mobilenet['val_loss'],
        mode='lines',
        name='MobileNetV2 Validation Loss',
        line=dict(color='blue', dash='dash')),
        row=1, col=2
    )
    fig.add_trace(go.Scatter(
        x=list(range(len(efficientnet['loss']))),
        y=efficientnet['loss'],
        mode='lines',
        name='EfficientNet Train Loss',
        line=dict(color='green')),
        row=1, col=2
    )
    fig.add_trace(go.Scatter(
        x=list(range(len(efficientnet['val_loss']))),
        y=efficientnet['val_loss'],
        mode='lines',
        name='EfficientNet Validation Loss',
        line=dict(color='green', dash='dash')),
        row=1, col=2
    )
    fig.add_trace(go.Scatter(
        x=list(range(len(vgg16['loss']))),
        y=vgg16['loss'],
        mode='lines',
        name='VGG16 Train Loss',
        line=dict(color='orange')),
        row=1, col=2
    )
    fig.add_trace(go.Scatter(
        x=list(range(len(vgg16['val_loss']))),
        y=vgg16['val_loss'],
        mode='lines',
        name='VGG16 Validation Loss',
        line=dict(color='orange', dash='dash')),
        row=1, col=2
    )

    # Update layout
    fig.update_layout(
        title="Comparison of Fine-Tuning Metrics Across Models",
        height=600,
        width=1200,
        xaxis_title="Epochs",
        yaxis_title="Metrics",
        template='plotly_dark',
        legend_title="Metrics",
        showlegend=True
    )

    # Show the figure
    fig.show()