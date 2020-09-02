from data_configs import (
    getConversations,
    preprocessSentence,
    fitTokenizerToCorpus,
    tokenizeData,
    saveTokenizer,
    padData,
    getTfDataset
)

from model_configs import (
    getTransformerModel,
    CustomSchedule,
    customLossFunction,
    accuracy,
    getCallbacks
)

from plot_visualizations import (
    plotTrainingHistory
)

import settings as s
from tensorflow.keras.optimizers import Adam

# Load text data
print('[INFO] Loading text data')
queries, responses = getConversations(
    'data/movie_lines.txt',
    'data/movie_conversations.txt',
    percent=s.DATASET_SIZE
)

# preprocess sentence
print('[INFO] Preprocessing data')
queries = list(map(preprocessSentence, queries))
responses = list(map(preprocessSentence, responses))

# fit tokenizer on corpus
print('[INFO] Fitting tokenizer on corpus')
tokenizer = fitTokenizerToCorpus(queries + responses, vocab_size=s.VOCAB_SIZE)

# save tokenizer to disk
saveTokenizer('models/myTokenizer.json', tokenizer)

# tokenize data
print('[INFO] Tokenizing data')
queries = tokenizeData(queries, tokenizer)
responses = tokenizeData(responses, tokenizer)

# pad sequences
print('[INFO] Padding sequences')
queries = padData(queries, maxlen=s.SEQUENCE_LENGTH)
responses = padData(responses, maxlen=s.SEQUENCE_LENGTH)

# get tf.data.Dataset
print('[INFO] Fetching tf.data.Dataset')
dataset = getTfDataset(
    queries, responses, batch_size=s.BATCH_SIZE,
    shuffer_buffer_size=s.SHUFFLE_BUFFER_SIZE
)

model = getTransformerModel(
    vocab_size=s.VOCAB_SIZE,
    num_layers=s.NUM_LAYERS,
    units=s.UNITS,
    d_model=s.D_MODEL,
    num_heads=s.NUM_HEADS,
    dropout=s.DROPOUT
)

model.summary()

# configure optimizer
learningRate = CustomSchedule(s.D_MODEL, warmup_steps=4000)
optimizer = Adam(learningRate, beta_1=0.9, beta_2=0.98, epsilon=1e-9)

# compile model with custom metrics
model.compile(
    optimizer=optimizer,
    loss=customLossFunction,
    metrics=[accuracy]
)

# train model with callbacks
callbacks = getCallbacks()
model.fit(dataset, epochs=s.EPOCHS, callbacks=callbacks)

print('[INFO] Saving model weights')
model.save_weights('models/final_model_weight')

print('[INFO] Plotting training history')
plotTrainingHistory('models/training_history.csv')
