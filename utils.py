import tensorflow as tf
import tensorflow_datasets as tfds

import train
import data_configs
from data_configs import preprocessSentence
from model_configs import getTransformerModel


def loadTrainedTransformerModel():
    model = getTransformerModel(
        vocab_size=data_configs.VOCAB_SIZE,
        num_layers=train.NUM_LAYERS,
        units=train.UNITS,
        d_model=train.D_MODEL,
        num_heads=train.NUM_HEADS,
        dropout=train.DROPOUT
    )

    model.load_weights('models/final_model_weight.hdf5')

    return model


def loadFitTokenizer():
    return tfds.features.text.SubwordTextEncoder.load_from_file('models/myTokenizer')


def evaluate(sentence, model, tokenizer):
    sentence = preprocessSentence(sentence)

    sentence = tf.expand_dims(
        data_configs.START_TOKEN + tokenizer.encode(sentence) + data_configs.END_TOKEN,
        axis=0
    )

    output = tf.expand_dims(data_configs.START_TOKEN, 0)

    for i in range(train.MAXLEN):
        predictions = model.predict([sentence, output])
        predictions = predictions[:, -1:, :]
        predictedID = tf.cast(tf.argmax(predictions, axis=-1), tf.int32)

        if tf.equal(predictedID, data_configs.END_TOKEN[0]):
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
