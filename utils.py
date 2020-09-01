import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow.keras.optimizers import Adam

import train
from data_configs import preprocessSentence

from model_configs import (
    getTransformerModel,
    CustomSchedule,
    customLossFunction,
    accuracy
)


def loadTrainedTransformerModel(path_to_model, tokenizer):
    model = getTransformerModel(
        vocab_size=tokenizer.vocab_size + 2,
        num_layers=train.NUM_LAYERS,
        units=train.UNITS,
        d_model=train.D_MODEL,
        num_heads=train.NUM_HEADS,
        dropout=train.DROPOUT
    )

    # configure optimizer
    learningRate = CustomSchedule(train.D_MODEL, warmup_steps=4000)
    optimizer = Adam(learningRate, beta_1=0.9, beta_2=0.98, epsilon=1e-9)

    # compile model with custom metrics
    model.compile(
        optimizer=optimizer,
        loss=customLossFunction,
        metrics=[accuracy]
    )

    model.load_weights(path_to_model)

    return model


def loadFitTokenizer(path_to_tokenizer):
    return tfds.features.text.SubwordTextEncoder.load_from_file(path_to_tokenizer)


def evaluate(sentence, model, tokenizer):
    sentence = preprocessSentence(sentence)

    sentence = tf.expand_dims(
        [tokenizer.vocab_size] + tokenizer.encode(sentence) + [tokenizer.vocab_size + 1],
        axis=0
    )

    output = tf.expand_dims([tokenizer.vocab_size], 0)

    for i in range(train.MAXLEN):
        predictions = model.predict([sentence, output])
        predictions = predictions[:, -1:, :]
        predictedID = tf.cast(tf.argmax(predictions, axis=-1), tf.int32)

        if tf.equal(predictedID, tokenizer.vocab_size + 1):
            break

        output = tf.concat([output, predictedID], axis=-1)

    return tf.squeeze(output, axis=0)


def predict(sentence, model, tokenizer):
    prediction = evaluate(sentence, model, tokenizer)

    predictedSentence = tokenizer.decode(
        [i for i in prediction if i < tokenizer.vocab_size]
    )

    print('=' * 85)
    print('Input:', sentence)
    print('Output:', predictedSentence)
    print()

    return predictedSentence
