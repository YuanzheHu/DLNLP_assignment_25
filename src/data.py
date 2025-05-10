import os
import pandas as pd
from typing import Tuple, List, Dict
import numpy as np
from sklearn.preprocessing import OneHotEncoder, LabelEncoder

def get_lines(filename: str) -> List[str]:
    """Read lines from a file."""
    with open(filename, "r") as f:
        return f.readlines()

def preprocess_text_with_line_numbers(filename: str) -> List[Dict]:
    """Preprocess text data and extract line numbers and targets."""
    input_lines = get_lines(filename)
    abstract_lines = ""
    abstract_samples = []
    
    for line in input_lines:
        if line.startswith("###"):
            # Start of new abstract
            abstract_lines = ""
        elif line.isspace():
            # End of abstract, process it
            abstract_line_split = abstract_lines.splitlines()
            
            for abstract_line_number, abstract_line in enumerate(abstract_line_split):
                target_text_split = abstract_line.split("\t")
                if len(target_text_split) != 2:
                    continue
                    
                line_data = {
                    "target": target_text_split[0],
                    "text": target_text_split[1].lower(),
                    "line_number": abstract_line_number,
                    "total_lines": len(abstract_line_split) - 1
                }
                abstract_samples.append(line_data)
        else:
            # Add line to current abstract
            abstract_lines += line
            
    return abstract_samples

def load_pubmed_rct_dataset(data_dir: str) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict]:
    """Load and preprocess the PubMed RCT dataset.
    
    Returns:
        Tuple containing:
        - train_df: Training data DataFrame
        - val_df: Validation data DataFrame
        - test_df: Test data DataFrame
        - label_info: Dictionary containing label encoder and class information
    """
    # Load and preprocess data
    train_samples = preprocess_text_with_line_numbers(os.path.join(data_dir, "train.txt"))
    val_samples = preprocess_text_with_line_numbers(os.path.join(data_dir, "dev.txt"))
    test_samples = preprocess_text_with_line_numbers(os.path.join(data_dir, "test.txt"))
    
    # Convert to DataFrames
    train_df = pd.DataFrame(train_samples)
    val_df = pd.DataFrame(val_samples)
    test_df = pd.DataFrame(test_samples)
    
    # Prepare label encoders
    label_encoder = LabelEncoder()
    one_hot_encoder = OneHotEncoder(sparse_output=False)
    
    # Fit label encoder on training data
    label_encoder.fit(train_df['target'])
    
    # Get class information
    num_classes = len(label_encoder.classes_)
    class_names = label_encoder.classes_.tolist()
    
    # Create label info dictionary
    label_info = {
        'label_encoder': label_encoder,
        'one_hot_encoder': one_hot_encoder,
        'num_classes': num_classes,
        'class_names': class_names
    }
    
    return train_df, val_df, test_df, label_info 