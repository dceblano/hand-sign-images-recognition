import plotly.graph_objs as go
import plotly.subplots as sp

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
