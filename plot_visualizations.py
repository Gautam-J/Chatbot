from model_configs import (
    PositionalEncoding,
    getEncoderLayer,
    getEncoderBlock,
    getDecoderLayer,
    getDecoderBlock,
    getTransformerModel,
    CustomSchedule
)

from data_configs import loadConversations

import seaborn as sns
import tensorflow as tf
import matplotlib.pyplot as plt

plt.style.use('seaborn')


def plotSamplePostionalEncoding():
    samplePosEncoding = PositionalEncoding(50, 512)

    plt.pcolormesh(samplePosEncoding.posEncoding.numpy()[0], cmap='RdBu')
    plt.xlabel('Depth')
    plt.xlim((0, 512))
    plt.ylabel('Position')
    plt.colorbar()
    plt.savefig('models/positionalEncoding.png')


def plotSampleEncoderLayer():
    sampleEncoderLayer = getEncoderLayer(
        units=512,
        d_model=128,
        num_heads=4,
        dropout=0.3,
        name="sample_encoder_layer"
    )

    tf.keras.utils.plot_model(
        sampleEncoderLayer, to_file='models/encoderLayer.png', show_shapes=True
    )


def plotSampleEncoderBlock():
    sampleEncoderBlock = getEncoderBlock(
        vocab_size=8192,
        num_layers=2,
        units=512,
        d_model=128,
        num_heads=4,
        dropout=0.3,
        name="sample_encoder"
    )

    tf.keras.utils.plot_model(
        sampleEncoderBlock, to_file='models/encoder.png', show_shapes=True
    )


def plotSampleDecoderLayer():
    sampleDecoderLayer = getDecoderLayer(
        units=512,
        d_model=128,
        num_heads=4,
        dropout=0.3,
        name="sample_decoder_layer"
    )

    tf.keras.utils.plot_model(
        sampleDecoderLayer, to_file='models/decoder_layer.png', show_shapes=True
    )


def plotSampleDecoderBlock():
    sampleDecoderBlock = getDecoderBlock(
        vocab_size=8192,
        num_layers=2,
        units=512,
        d_model=128,
        num_heads=4,
        dropout=0.3,
        name="sample_decoder"
    )

    tf.keras.utils.plot_model(
        sampleDecoderBlock, to_file='models/decoder.png', show_shapes=True
    )


def plotSampleTransformerModel():
    sampleTransformerModel = getTransformerModel(
        vocab_size=8192,
        num_layers=4,
        units=512,
        d_model=128,
        num_heads=4,
        dropout=0.3,
        name="sample_transformer"
    )

    tf.keras.utils.plot_model(
        sampleTransformerModel, to_file='models/transformer.png', show_shapes=True
    )


def plotSampleLearningRateSchedule():
    sampleLearningRate = CustomSchedule(d_model=128)

    plt.plot(sampleLearningRate(tf.range(200000, dtype=tf.float32)))
    plt.ylabel("Learning Rate")
    plt.xlabel("Train Step")
    plt.savefig('models/learningRateSchedule.png')


def plotSequenceLengthHistogram(corpus):
    sequenceLengths = [len(sentence.split()) for sentence in corpus]

    sns.distplot(sequenceLengths, bins=100)
    plt.xlim(-10, 100)
    plt.savefig('models/sequenceLengths.png')


def plotTrainingHistory(history):
    pass


if __name__ == '__main__':
    plotSamplePostionalEncoding()
    plotSampleEncoderLayer()
    plotSampleEncoderBlock()
    plotSampleDecoderLayer()
    plotSampleDecoderBlock()
    plotSampleTransformerModel()
    plotSampleLearningRateSchedule()
