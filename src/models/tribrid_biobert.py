import tensorflow as tf
from tensorflow.keras import layers
import tensorflow_hub as hub

def build_tribrid_biobert_model(num_classes: int) -> tf.keras.Model:
    # Text input
    text_input = layers.Input(shape=[], dtype=tf.string, name="text_input")
    
    # BioBERT preprocessing
    preprocessor = hub.KerasLayer(
        "https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3",
        name="preprocessor"
    )
    
    # BioBERT encoder
    biobert_layer = hub.KerasLayer(
        "https://tfhub.dev/google/experts/bert/pubmed/2",
        trainable=False,
        name="biobert_layer"
    )
    
    # Process text through BioBERT
    preprocessed = preprocessor(text_input)
    biobert_outputs = biobert_layer(preprocessed)
    
    # Get pooled output (CLS token)
    pooled_output = biobert_outputs["pooled_output"]
    
    # Add classification head
    x = layers.Dense(256, activation="relu")(pooled_output)
    x = layers.Dropout(0.5)(x)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(0.5)(x)
    output_layer = layers.Dense(num_classes, activation="softmax")(x)
    
    model = tf.keras.Model(
        inputs=text_input,
        outputs=output_layer,
        name="tribrid_biobert"
    )
    return model 