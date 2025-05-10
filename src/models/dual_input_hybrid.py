import tensorflow as tf
from tensorflow.keras import layers
import tensorflow_hub as hub

def build_dual_input_hybrid_model(num_classes: int, output_seq_char_len: int, num_char_tokens: int) -> tf.keras.Model:
    # Token input (USE)
    token_inputs = layers.Input(shape=[], dtype=tf.string, name="token_input")
    tf_hub_embedding_layer = hub.KerasLayer(
        "https://tfhub.dev/google/universal-sentence-encoder/4",
        trainable=False,
        name="universal_sentence_encoder"
    )
    token_embeddings = tf_hub_embedding_layer(token_inputs)
    token_output = layers.Dense(128, activation="relu")(token_embeddings)
    token_model = tf.keras.Model(inputs=token_inputs, outputs=token_output)

    # Char input
    char_inputs = layers.Input(shape=(1,), dtype=tf.string, name="char_input")
    char_vectorizer = layers.TextVectorization(
        max_tokens=num_char_tokens,
        output_sequence_length=output_seq_char_len,
        standardize="lower_and_strip_punctuation",
        name="char_vectorizer"
    )
    # Note: char_vectorizer.adapt() must be called outside this function with training data
    char_embed = layers.Embedding(
        input_dim=num_char_tokens,
        output_dim=25,
        mask_zero=False,
        name="char_embed"
    )
    char_vectors = char_vectorizer(char_inputs)
    char_embeddings = char_embed(char_vectors)
    char_bi_lstm = layers.Bidirectional(layers.LSTM(25))(char_embeddings)
    char_model = tf.keras.Model(inputs=char_inputs, outputs=char_bi_lstm)

    # Concatenate
    token_char_concat = layers.Concatenate(name="token_char_hybrid")([
        token_model.output, char_model.output
    ])
    combined_dropout = layers.Dropout(0.5)(token_char_concat)
    combined_dense = layers.Dense(200, activation="relu")(combined_dropout)
    final_dropout = layers.Dropout(0.5)(combined_dense)
    output_layer = layers.Dense(num_classes, activation="softmax")(final_dropout)

    model = tf.keras.Model(
        inputs=[token_model.input, char_model.input],
        outputs=output_layer,
        name="dual_input_hybrid"
    )
    return model 