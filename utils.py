import tensorflow as tf
from tensorflow_datasets.features.text import SubwordTextEncoder

from model_configs import getTransformerModel
from data_configs import preprocessSentence, START_TOKEN, END_TOKEN, maxlen


def loadTrainedTransformerModel():
    model = getTransformerModel()
    model.load_weights('models/modelWeights.hdf5')

    return model


def loadFitTokenizer():
    return SubwordTextEncoder.load_from_file('models/myTokenizer')


def evaluate(sentence, model, tokenizer):
    sentence = preprocessSentence(sentence)

    sentence = tf.expand_dims(
        START_TOKEN + tokenizer.encode(sentence) + END_TOKEN,
        axis=0
    )

    output = tf.expand_dims(START_TOKEN, 0)

    for i in range(maxlen):
        predictions = model.predict([sentence, output])
        predictions = predictions[:, -1:, :]
        predictedID = tf.cast(tf.argmax(predictions, axis=-1), tf.int32)

        if tf.equal(predictedID, END_TOKEN[0]):
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
