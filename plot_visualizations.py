from model_configs import (
    PositionalEncoding,
    getEncoderLayer,
    getEncoderBlock,
    getDecoderLayer,
    getDecoderBlock,
    getTransformerModel,
    CustomSchedule
)

import seaborn as sns
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt

plt.style.use('seaborn')


def plotSamplePostionalEncoding():
    samplePosEncoding = PositionalEncoding(50, 256)

    plt.pcolormesh(samplePosEncoding.posEncoding.numpy()[0], cmap='RdBu')
    plt.xlabel('Depth')
    plt.xlim((0, 256))
    plt.ylabel('Position')
    plt.colorbar()
    plt.savefig('models/positionalEncoding.png')
    plt.close()


def plotSampleEncoderLayer():
    sampleEncoderLayer = getEncoderLayer(
        units=512,
        d_model=256,
        num_heads=8,
        dropout=0.1,
        name="sample_encoder_layer"
    )

    tf.keras.utils.plot_model(
        sampleEncoderLayer, to_file='models/encoderLayer.png', show_shapes=True
    )


def plotSampleEncoderBlock():
    sampleEncoderBlock = getEncoderBlock(
        vocab_size=8226,
        num_layers=2,
        units=512,
        d_model=256,
        num_heads=8,
        dropout=0.1,
        name="sample_encoder"
    )

    tf.keras.utils.plot_model(
        sampleEncoderBlock, to_file='models/encoderBlock.png', show_shapes=True
    )


def plotSampleDecoderLayer():
    sampleDecoderLayer = getDecoderLayer(
        units=512,
        d_model=256,
        num_heads=8,
        dropout=0.1,
        name="sample_decoder_layer"
    )

    tf.keras.utils.plot_model(
        sampleDecoderLayer, to_file='models/decoderLayer.png', show_shapes=True
    )


def plotSampleDecoderBlock():
    sampleDecoderBlock = getDecoderBlock(
        vocab_size=8226,
        num_layers=2,
        units=512,
        d_model=256,
        num_heads=8,
        dropout=0.1,
        name="sample_decoder"
    )

    tf.keras.utils.plot_model(
        sampleDecoderBlock, to_file='models/decoderBlock.png', show_shapes=True
    )


def plotSampleTransformerModel():
    sampleTransformerModel = getTransformerModel(
        vocab_size=8226,
        num_layers=2,
        units=512,
        d_model=256,
        num_heads=8,
        dropout=0.1,
        name="sample_transformer"
    )

    tf.keras.utils.plot_model(
        sampleTransformerModel, to_file='models/transformer.png', show_shapes=True
    )


def plotSampleLearningRateSchedule():
    sampleLearningRate = CustomSchedule(d_model=256)

    plt.plot(sampleLearningRate(tf.range(100000, dtype=tf.float32)))
    plt.ylabel("Learning Rate")
    plt.xlabel("Train Step")
    plt.savefig('models/learningRateSchedule.png')
    plt.close()


def plotSequenceLengthHistogram(corpus):
    sequenceLengths = [len(sentence) for sentence in corpus]

    sns.distplot(sequenceLengths, bins=100)
    plt.xlim(-10, 100)
    plt.savefig('models/sequenceLengths.png')
    plt.close()


def plotTrainingHistory(path_to_csv_logger_file):
    df = pd.read_csv(path_to_csv_logger_file)

    for column in df.columns:
        if not column == 'epoch':
            plt.plot(df['epoch'], df[column], label=column)
            plt.title(column)
            plt.xlabel('Epoch')
            plt.ylabel(column)
            plt.legend()
            plt.savefig(f'models/training_history_{column}.png')
            plt.close()


if __name__ == '__main__':
    plotSamplePostionalEncoding()
    plotSampleEncoderLayer()
    plotSampleEncoderBlock()
    plotSampleDecoderLayer()
    plotSampleDecoderBlock()
    plotSampleTransformerModel()
    plotSampleLearningRateSchedule()
