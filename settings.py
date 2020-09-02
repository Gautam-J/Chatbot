# Data preprocessing
VOCAB_SIZE = 8000
SEQUENCE_LENGTH = 30
DATASET_SIZE = 0.22595601992028272  # percentage of total data used for training
BATCH_SIZE = 64
SHUFFLE_BUFFER_SIZE = 25000

# token tags
STARTOFSENTENCE_TOKEN = '<sos>'
ENDOFSENTENCE_TOKEN = '<eos>'
OUTOFVOCABULARY_TOKEN = '<oov>'

# Hyperparameters
DROPOUT = 0.1
EPOCHS = 20
NUM_LAYERS = 2
UNITS = 512
D_MODEL = 256  # d_model % num_heads == 0
NUM_HEADS = 8
