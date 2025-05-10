import os
import numpy as np
import tensorflow as tf
from typing import Dict, List, Tuple
from .data import load_pubmed_rct_dataset
from .utils import calculate_metrics, plot_confusion_matrix, plot_training_history, save_metrics
from .utils import split_chars

def evaluate_model(model: tf.keras.Model, 
                  data_dir: str,
                  model_name: str,
                  output_dir: str = 'evaluation_results') -> Dict:
    """Evaluate model on test data and save results."""
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Load test data and label information
    _, _, test_df, label_info = load_pubmed_rct_dataset(data_dir)
    test_sentences = test_df['text'].tolist()
    label_encoder = label_info['label_encoder']
    class_names = label_info['class_names']
    
    # Prepare test data based on model type
    if model_name == "dual_input_hybrid":
        test_chars = [split_chars(s) for s in test_sentences]
        test_data = tf.data.Dataset.from_tensor_slices((test_sentences, test_chars))
    else:
        test_data = tf.data.Dataset.from_tensor_slices(test_sentences)
    
    # Batch and prefetch
    test_data = test_data.batch(32).prefetch(tf.data.AUTOTUNE)
    
    # Get predictions
    predictions = model.predict(test_data)
    y_pred = np.argmax(predictions, axis=1)
    y_true = label_encoder.transform(test_df['target'].values)
    
    # Calculate metrics
    metrics = calculate_metrics(y_true, y_pred, class_names)
    
    # Save results
    model_output_dir = os.path.join(output_dir, model_name)
    os.makedirs(model_output_dir, exist_ok=True)
    
    # Save metrics
    save_metrics(metrics, os.path.join(model_output_dir, 'metrics.txt'))
    
    # Plot confusion matrix
    plot_confusion_matrix(
        metrics['confusion_matrix'],
        class_names,
        os.path.join(model_output_dir, 'confusion_matrix.png')
    )
    
    return metrics

def compare_models(models: Dict[str, tf.keras.Model],
                  data_dir: str,
                  output_dir: str = 'evaluation_results') -> Dict[str, Dict]:
    """Compare multiple models and save results."""
    results = {}
    
    # Load label information once
    _, _, _, label_info = load_pubmed_rct_dataset(data_dir)
    class_names = label_info['class_names']
    
    for model_name, model in models.items():
        print(f"Evaluating {model_name}...")
        results[model_name] = evaluate_model(
            model=model,
            data_dir=data_dir,
            model_name=model_name,
            output_dir=output_dir
        )
    
    # Save comparison summary
    summary_path = os.path.join(output_dir, 'model_comparison.txt')
    with open(summary_path, 'w') as f:
        f.write("Model Comparison Summary\n")
        f.write("======================\n\n")
        
        for model_name, metrics in results.items():
            f.write(f"{model_name}:\n")
            f.write("-------------------\n")
            report = metrics['classification_report']
            
            # Write overall metrics
            f.write("Overall Metrics:\n")
            for metric in ['accuracy', 'macro avg', 'weighted avg']:
                if metric in report:
                    f.write(f"{metric}:\n")
                    for score_name, score in report[metric].items():
                        if isinstance(score, float):
                            f.write(f"  {score_name}: {score:.4f}\n")
            f.write("\n")
    
    return results 