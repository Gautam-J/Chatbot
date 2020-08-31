from data_configs import (
    loadConversations,
    getTfDataset,
    fitTokenizerToCorpus,
    tokenizeData,
    padTokenizedData
)

MAXLEN = 40
BATCH_SIZE = 32
SHUFFLE_BUFFER_SIZE = 100000

# Load text data
print('[INFO] Loading text data')
queries, responses = loadConversations()

# fit tokenizer on corpus
print('[INFO] Fitting tokenizer on corpus')
tokenizer = fitTokenizerToCorpus(queries + responses, target_vocab_size=2**13)

# tokenize data
print('[INFO] Tokenizing data')
queries = tokenizeData(queries, tokenizer)
responses = tokenizeData(responses, tokenizer)

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
