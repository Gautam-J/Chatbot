# Data preprocessing
VOCAB_SIZE = 8000
SEQUENCE_LENGTH = 20
DATASET_SIZE = 0.2  # percentage of total data used for training

# token tags
STARTOFSENTENCE_TOKEN = '<sos>'
ENDOFSENTENCE_TOKEN = '<eos>'
OUTOFVOCABULARY_TOKEN = '<oov>'

# Hyperparameters
BATCH_SIZE = 32
SHUFFLE_BUFFER_SIZE = 10000
EPOCHS = 15
NUM_LAYERS = 2
NUM_HEADS = 8
D_MODEL = 128
UNITS = 512
DROPOUT = 0.1
