import tensorflow as tf
from tensorflow.keras import layers
import tensorflow_hub as hub

def build_tribrid_use_model(num_classes: int) -> tf.keras.Model:
    # Text input
    text_input = layers.Input(shape=[], dtype=tf.string, name="text_input")
    
    # Universal Sentence Encoder
    use_layer = hub.KerasLayer(
        "https://tfhub.dev/google/universal-sentence-encoder/4",
        trainable=False,
        name="universal_sentence_encoder"
    )
    
    # Get embeddings
    embeddings = use_layer(text_input)
    
    # Add dense layers
    x = layers.Dense(256, activation="relu")(embeddings)
    x = layers.Dropout(0.5)(x)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(0.5)(x)
    output_layer = layers.Dense(num_classes, activation="softmax")(x)
    
    model = tf.keras.Model(
        inputs=text_input,
        outputs=output_layer,
        name="tribrid_use"
    )
    return model 