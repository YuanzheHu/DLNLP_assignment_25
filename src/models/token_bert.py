import tensorflow as tf
from tensorflow.keras import layers
import tensorflow_hub as hub

def build_token_bert_model(num_classes: int) -> tf.keras.Model:
    # BERT input
    text_input = layers.Input(shape=[], dtype=tf.string, name="text_input")
    
    # Load BERT layer from TF Hub
    bert_layer = hub.KerasLayer(
        "https://tfhub.dev/tensorflow/bert_en_uncased_L-12_H-768_A-12/4",
        trainable=False,
        name="bert_layer"
    )
    
    # Preprocessing
    preprocessor = hub.KerasLayer(
        "https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3",
        name="preprocessor"
    )
    
    # Process text through BERT
    preprocessed = preprocessor(text_input)
    bert_outputs = bert_layer(preprocessed)
    
    # Get pooled output (CLS token)
    pooled_output = bert_outputs["pooled_output"]
    
    # Add classification head
    x = layers.Dense(128, activation="relu")(pooled_output)
    x = layers.Dropout(0.5)(x)
    output_layer = layers.Dense(num_classes, activation="softmax")(x)
    
    model = tf.keras.Model(
        inputs=text_input,
        outputs=output_layer,
        name="token_bert"
    )
    return model 