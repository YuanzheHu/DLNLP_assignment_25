from typing import List, Dict, Tuple
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

def split_chars(text: str) -> str:
    """Split text into characters with spaces."""
    return " ".join(list(text))

def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray, class_names: List[str]) -> Dict:
    """Calculate classification metrics."""
    report = classification_report(y_true, y_pred, target_names=class_names, output_dict=True)
    conf_matrix = confusion_matrix(y_true, y_pred)
    return {
        'classification_report': report,
        'confusion_matrix': conf_matrix
    }

def plot_confusion_matrix(conf_matrix: np.ndarray, class_names: List[str], save_path: str = None):
    """Plot confusion matrix."""
    plt.figure(figsize=(10, 8))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    if save_path:
        plt.savefig(save_path)
    plt.close()

def plot_training_history(history: Dict, save_path: str = None):
    """Plot training and validation metrics."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Plot accuracy
    ax1.plot(history['accuracy'], label='Training Accuracy')
    ax1.plot(history['val_accuracy'], label='Validation Accuracy')
    ax1.set_title('Model Accuracy')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Accuracy')
    ax1.legend()
    
    # Plot loss
    ax2.plot(history['loss'], label='Training Loss')
    ax2.plot(history['val_loss'], label='Validation Loss')
    ax2.set_title('Model Loss')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.legend()
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    plt.close()

def save_metrics(metrics: Dict, save_path: str):
    """Save metrics to a text file."""
    with open(save_path, 'w') as f:
        f.write("Classification Report:\n")
        f.write("===================\n")
        for class_name, scores in metrics['classification_report'].items():
            if isinstance(scores, dict):
                f.write(f"\n{class_name}:\n")
                for metric, value in scores.items():
                    f.write(f"{metric}: {value:.4f}\n")
            else:
                f.write(f"{class_name}: {scores:.4f}\n")
                
def save_model(model: tf.keras.Model, save_path: str):
    """Save model to a file."""
    model.save(save_path)