import tensorflow as tf
from tensorflow.keras.optimizers import Adam

from data_configs import preprocessSentence, padData

from model_configs import (
    getTransformerModel,
    CustomSchedule,
    customLossFunction,
    accuracy
)

import settings as s


def loadTrainedTransformerModel(path_to_model):
    model = getTransformerModel(
        vocab_size=s.VOCAB_SIZE,
        num_layers=s.NUM_LAYERS,
        units=s.UNITS,
        d_model=s.D_MODEL,
        num_heads=s.NUM_HEADS,
        dropout=s.DROPOUT
    )

    # configure optimizer
    learningRate = CustomSchedule(s.D_MODEL, warmup_steps=4000)
    optimizer = Adam(learningRate, beta_1=0.9, beta_2=0.98, epsilon=1e-9)

    # compile model with custom metrics
    model.compile(
        optimizer=optimizer,
        loss=customLossFunction,
        metrics=[accuracy]
    )

    model.load_weights(path_to_model)

    return model


def evaluate(sentence, model, tokenizer):
    sentence = preprocessSentence(sentence)
    sentence = tokenizer.texts_to_sequences([sentence])
    sentence = padData(sentence, s.SEQUENCE_LENGTH)

    output = tf.expand_dims([tokenizer.word_index[s.STARTOFSENTENCE_TOKEN]], 0)

    for i in range(s.SEQUENCE_LENGTH):
        predictions = model([sentence, output], training=False)
        predictions = predictions[:, -1:, :]
        predictedID = tf.cast(tf.argmax(predictions, axis=-1), tf.int32)

        if tf.equal(predictedID, tokenizer.word_index[s.ENDOFSENTENCE_TOKEN]):
            break

        output = tf.concat([output, predictedID], axis=-1)

    return tf.squeeze(output, axis=0)


def predict(sentence, model, tokenizer):
    prediction = evaluate(sentence, model, tokenizer)
    prediction = tokenizer.sequences_to_texts([prediction.numpy()])
    predictedSentence = prediction[0].lstrip(s.STARTOFSENTENCE_TOKEN + ' ')

    return predictedSentence
