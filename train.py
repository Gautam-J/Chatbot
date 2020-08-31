from data_configs import (
    loadConversations,
    getTfDataset,
    fitTokenizerToCorpus,
    tokenizeData,
    padTokenizedData
)

from model_configs import (
    getTransformerModel,
    CustomSchedule,
    customLossFunction,
    customAccuracyMetric,
    getCallbacks
)

from plot_visualizations import (
    plotSequenceLengthHistogram,
    plotTrainingHistory
)

import data_configs
from tensorflow.keras.optimizers import Adam

MAXLEN = 40
BATCH_SIZE = 32
SHUFFLE_BUFFER_SIZE = 100000

EPOCHS = 5
NUM_LAYERS = 2
D_MODEL = 256
NUM_HEADS = 8
UNITS = 512
DROPOUT = 0.1


def main():
    # Load text data
    print('[INFO] Loading text data')
    queries, responses = loadConversations()

    # using only 1000 datapoints for faster training
    queries, responses = queries[:1000], responses[:1000]

    # fit tokenizer on corpus
    print('[INFO] Fitting tokenizer on corpus')
    tokenizer = fitTokenizerToCorpus(queries + responses)

    # tokenize data
    print('[INFO] Tokenizing data')
    queries = tokenizeData(queries, tokenizer)
    responses = tokenizeData(responses, tokenizer)

    print('[INFO] Plotting KDE for sequence length')
    plotSequenceLengthHistogram(queries + responses)

    # pad sequences
    print('[INFO] Padding sequences')
    queries = padTokenizedData(queries, maxlen=MAXLEN)
    responses = padTokenizedData(responses, maxlen=MAXLEN)

    # get tf.data.Dataset
    print('[INFO] Fetching tf.data.Dataset')
    dataset = getTfDataset(
        queries, responses, batch_size=BATCH_SIZE,
        shuffer_buffer_size=SHUFFLE_BUFFER_SIZE
    )

    print(dataset)

    model = getTransformerModel(
        vocab_size=data_configs.VOCAB_SIZE,
        num_layers=NUM_LAYERS,
        units=UNITS,
        d_model=D_MODEL,
        num_heads=NUM_HEADS,
        dropout=DROPOUT
    )

    model.summary()

    # configure optimizer
    learningRate = CustomSchedule(D_MODEL, warmup_steps=4000)
    optimizer = Adam(learningRate, beta_1=0.9, beta_2=0.98, epsilon=1e-9)

    # compile model with custom metrics
    model.compile(
        optimizer=optimizer,
        loss=customLossFunction,
        metrics=[customAccuracyMetric]
    )

    # train model with callbacks
    callbacks = getCallbacks()
    model.fit(dataset, epochs=EPOCHS, callbacks=callbacks)

    print('[INFO] Saving model weights')
    model.save_weights('models/final_model_weight.hdf5')

    print('[INFO] Plotting training history')
    plotTrainingHistory('models/training_history.csv')


if __name__ == '__main__':
    main()
