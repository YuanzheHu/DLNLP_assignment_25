import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import callbacks
from .models.dual_input_hybrid import build_dual_input_hybrid_model
from .models.token_bert import build_token_bert_model
from .models.tribrid_use import build_tribrid_use_model
from .models.tribrid_biobert import build_tribrid_biobert_model
from .utils import split_chars, plot_training_history
from .data import load_pubmed_rct_dataset
from .evaluate import evaluate_model

# Constants for char-level processing
import string
ALPHABET = string.ascii_lowercase + string.digits + string.punctuation
OUTPUT_SEQ_CHAR_LEN = 290
NUM_CHAR_TOKENS = len(ALPHABET) + 2  # alphabet + space + OOV

def train_model(model_name: str, 
                data_dir: str, 
                epochs: int = 5, 
                batch_size: int = 32, 
                checkpoint_path: str = 'best_weights/checkpoint.ckpt',
                output_dir: str = 'training_results') -> Tuple[tf.keras.Model, Dict]:
    """Train a model and return the trained model and training history."""
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    model_output_dir = os.path.join(output_dir, model_name)
    os.makedirs(model_output_dir, exist_ok=True)

    # Load data and label information
    train_df, val_df, test_df, label_info = load_pubmed_rct_dataset(data_dir)
    num_classes = label_info['num_classes']
    label_encoder = label_info['label_encoder']
    one_hot_encoder = label_info['one_hot_encoder']

    # Prepare data
    train_sentences = train_df['text'].tolist()
    val_sentences = val_df['text'].tolist()
    
    # Prepare labels
    train_labels = one_hot_encoder.fit_transform(train_df['target'].to_numpy().reshape(-1, 1))
    val_labels = one_hot_encoder.transform(val_df['target'].to_numpy().reshape(-1, 1))

    # Build model based on model_name
    if model_name == "dual_input_hybrid":
        train_chars = [split_chars(s) for s in train_sentences]
        val_chars = [split_chars(s) for s in val_sentences]
        
        # Build char vectorizer and adapt
        char_vectorizer = tf.keras.layers.TextVectorization(
            max_tokens=NUM_CHAR_TOKENS,
            output_sequence_length=OUTPUT_SEQ_CHAR_LEN,
            standardize="lower_and_strip_punctuation",
            name="char_vectorizer"
        )
        char_vectorizer.adapt(train_chars)
        
        model = build_dual_input_hybrid_model(num_classes, OUTPUT_SEQ_CHAR_LEN, NUM_CHAR_TOKENS)
        model.get_layer('char_vectorizer').set_vocabulary(char_vectorizer.get_vocabulary())
        
        # Prepare tf.data datasets for dual input
        train_data = tf.data.Dataset.from_tensor_slices(((train_sentences, train_chars), train_labels))
        val_data = tf.data.Dataset.from_tensor_slices(((val_sentences, val_chars), val_labels))
    else:
        if model_name == "token_bert":
            model = build_token_bert_model(num_classes)
        elif model_name == "tribrid_use":
            model = build_tribrid_use_model(num_classes)
        elif model_name == "tribrid_biobert":
            model = build_tribrid_biobert_model(num_classes)
        else:
            raise ValueError(f"Unknown model name: {model_name}")
        
        # Prepare tf.data datasets for single input
        train_data = tf.data.Dataset.from_tensor_slices((train_sentences, train_labels))
        val_data = tf.data.Dataset.from_tensor_slices((val_sentences, val_labels))

    # Compile model
    model.compile(
        loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.2),
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
        metrics=['accuracy']
    )

    # Prepare datasets
    train_data = train_data.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    val_data = val_data.batch(batch_size).prefetch(tf.data.AUTOTUNE)

    # Callbacks
    os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
    model_checkpoint_callback = callbacks.ModelCheckpoint(
        filepath=checkpoint_path,
        save_weights_only=True,
        save_best_only=True,
        monitor='val_loss',
        save_freq='epoch'
    )
    early_stopping = callbacks.EarlyStopping(
        monitor='val_loss',
        patience=3,
        min_delta=0.5,
        verbose=1,
        restore_best_weights=True
    )
    reduce_lr = callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.2,
        patience=2,
        verbose=1,
        min_lr=1e-7
    )
    tensorboard_callback = callbacks.TensorBoard(
        log_dir=os.path.join(model_output_dir, 'logs'),
        histogram_freq=1
    )

    # Train
    history = model.fit(
        train_data,
        epochs=epochs,
        validation_data=val_data,
        callbacks=[
            model_checkpoint_callback,
            early_stopping,
            reduce_lr,
            tensorboard_callback
        ]
    )

    # Plot training history
    plot_training_history(
        history.history,
        os.path.join(model_output_dir, 'training_history.png')
    )

    # Evaluate model
    print(f"\nEvaluating {model_name} model...")
    metrics = evaluate_model(
        model=model,
        data_dir=data_dir,
        model_name=model_name,
        output_dir=output_dir
    )

    return model, history.history 