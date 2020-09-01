import re
import json
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer, tokenizer_from_json

import settings as s


def loadLines(path_to_lines):
    fields = ["lineID", "characterID", "movieID", "character", "text"]

    lines = {}
    with open(path_to_lines, 'r', encoding='iso-8859-1') as f:
        for line in f:
            values = line.split(' +++$+++ ')

            lineObject = {}
            for i, field in enumerate(fields):
                lineObject[field] = values[i]

            lines[lineObject['lineID']] = lineObject

    return lines


def loadConversations(path_to_conversations, lines):
    fields = [
        "character1ID",
        "character2ID",
        "movieID",
        "utteranceIDs"
    ]

    conversations = []
    with open(path_to_conversations, 'r', encoding='iso-8859-1') as f:
        for line in f:
            values = line.split(' +++$+++ ')

            convObject = {}
            for i, field in enumerate(fields):
                convObject[field] = values[i]

            # convert string to list
            utteranceIdPattern = re.compile('L[0-9]+')
            lineIds = utteranceIdPattern.findall(convObject['utteranceIDs'])

            # reassemble lines
            convObject['lines'] = []
            for lineId in lineIds:
                convObject['lines'].append(lines[lineId])

            conversations.append(convObject)

    return conversations


def getConversations(path_to_lines, path_to_conversations):
    lines = loadLines(path_to_lines)
    conversations = loadConversations(path_to_conversations, lines)

    pairs = []
    for conv in conversations:
        for i in range(len(conv['lines']) - 1):
            query = conv['lines'][i]['text'].strip()
            response = conv['lines'][i + 1]['text'].strip()

            if query and response:
                pairs.append([query, response])

    return zip(*pairs)


def preprocessSentence(sentence):
    # lower case all letters and remove whitespace
    sentence = sentence.lower().strip()

    # add space before and after punctuations r"([<punctuations>])"
    sentence = re.sub(r"([?.!,])", r" \1 ", sentence)

    # add space for any char outside r"[<allowedCharacters>]"
    sentence = re.sub(r"[^a-zA-Z?.!,]+", " ", sentence)

    # remove extra spaces, only one space character allowed
    sentence = re.sub(r'[" "]+', " ", sentence)

    # remove whitespace again
    sentence = sentence.strip()

    # add tags in the beginning and end
    sentence = s.STARTOFSENTENCE_TOKEN + ' ' + sentence + ' ' + s.ENDOFSENTENCE_TOKEN

    return sentence


def fitTokenizerToCorpus(corpus, vocab_size=None):
    # initialize tokenizer with no filters, vocab_size, and out of vocabulary token
    tokenizer = Tokenizer(filters='', num_words=vocab_size, oov_token='<OOV>')

    # fit on data
    tokenizer.fit_on_texts(corpus)

    return tokenizer


def saveTokenizer(path_to_file, tokenizer):
    tk_json = tokenizer.to_json()

    with open(path_to_file, 'w') as f:
        json.dump(tk_json, f)

    print('[INFO] Saved tokenizer')


def loadTokenizer(path_to_file):

    with open(path_to_file) as f:
        tk_json = json.load(f)

    tokenizer = tokenizer_from_json(tk_json)

    return tokenizer


def tokenizeData(data, tokenizer):
    return tokenizer.texts_to_sequences(data)


def padData(data, maxlen=None):
    return pad_sequences(data, maxlen=maxlen, padding='post', truncating='post')


def getTfDataset(inputs, outputs, batch_size=32, shuffer_buffer_size=100000):

    dataset = tf.data.Dataset.from_tensor_slices((
        {'inputs': inputs,
         'dec_inputs': outputs[:, :-1]},
        {'outputs': outputs[:, 1:]}
    ))

    dataset = dataset.shuffle(shuffer_buffer_size)
    dataset = dataset.batch(batch_size)
    dataset = dataset.cache()
    dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)

    return dataset
