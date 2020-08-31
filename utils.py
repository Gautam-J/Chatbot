import tensorflow as tf
import tensorflow_datasets as tfds

import train
from data_configs import preprocessSentence
from model_configs import getTransformerModel


def loadTrainedTransformerModel(path_to_model, tokenizer):
    model = getTransformerModel(
        vocab_size=tokenizer.vocab_size + 2,
        num_layers=train.NUM_LAYERS,
        units=train.UNITS,
        d_model=train.D_MODEL,
        num_heads=train.NUM_HEADS,
        dropout=train.DROPOUT
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
