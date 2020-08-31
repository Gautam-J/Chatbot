import os
import re
import tensorflow as tf
from tensorflow_datasets.features.text import SubwordTextEncoder
from tensorflow.keras.preprocessing.sequence import pad_sequences


def preprocessSentence(sentence):
    # lowercase all and remove whitespace
    sentence = sentence.lower().strip()

    # add space between punctuation and word
    sentence = re.sub(r"([?.!,])", r" \1", sentence)
    sentence = re.sub(r'[" "]+', " ", sentence)

    # replace everything with space except (a-z, A-Z, ".", "?", "!", ",")
    sentence = re.sub(r"[^a-zA-Z?.!,]+", " ", sentence)
    sentence = sentence.strip()

    return sentence


def getIdToLineDictionary():

    pathToMovieLines = os.path.join('data', 'movie_lines.txt')

    # dictionary of line id to text
    idToLine = {}

    with open(pathToMovieLines, errors='ignore') as f:
        lines = f.readlines()

    for line in lines:
        parts = line.replace('\n', '').split(' +++$+++ ')
        idToLine[parts[0]] = parts[4]

    return idToLine


def loadConversations():

    pathToMovieConversations = os.path.join('data', 'movie_conversations.txt')
    idToLine = getIdToLineDictionary()
    queries, responses = [], []

    with open(pathToMovieConversations) as f:
        lines = f.readlines()

    for line in lines:
        parts = line.replace('\n', '').split(' +++$+++ ')
        conversation = [line[1:-1] for line in parts[3][1:-1].split(', ')]

        for i in range(len(conversation) - 1):
            queries.append(preprocessSentence(idToLine[conversation[i]]))
            responses.append(preprocessSentence(idToLine[conversation[i + 1]]))

    return queries, responses


def fitTokenizerToCorpus(corpus, target_vocab_size=2**13):

    global START_TOKEN, END_TOKEN, VOCAB_SIZE

    tokenizer = SubwordTextEncoder.build_from_corpus(
        corpus, target_vocab_size=target_vocab_size
    )

    START_TOKEN, END_TOKEN = [tokenizer.vocab_size], [tokenizer.vocab_size + 1]
    VOCAB_SIZE = tokenizer.vocab_size + 2

    tokenizer.save_to_file('myTokenizer')

    return tokenizer


def tokenizeData(data, tokenizer):

    tokenizedData = []
    for sentence in data:
        sentence = START_TOKEN + tokenizer.encode(sentence) + END_TOKEN
        tokenizedData.append(sentence)

    return tokenizedData


def padTokenizedData(data, maxlen=40):
    global maxlen

    paddedData = pad_sequences(data, maxlen=maxlen, padding='post',
                               truncating='post')

    return paddedData


def getTfDataset(inputs, outputs, batch_size=32, shuffer_buffer_size=1e5):

    dataset = tf.data.Dataset.from_tensor_slices((
        {'inputs': inputs,
         'dec_inputs': outputs[:, :-1]},
        {'outputs': outputs[:, 1:]}
    ))

    dataset = dataset.cache()
    dataset = dataset.shuffle(shuffer_buffer_size)
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)

    return dataset
