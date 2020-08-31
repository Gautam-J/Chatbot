import tensorflow as tf
from data_configs import maxlen


def getScaledDotProductAttention(query, key, value, mask):
    # calculate dot product between query and key
    matmulQK = tf.matmul(query, key, transpose_b=True)

    # scale
    depth = tf.cast(tf.shape(key)[-1], tf.float32)
    logits = matmulQK / tf.math.sqrt(depth)

    # set -Inf (softmax >> zero) for padding tokens
    if mask is not None:
        logits += (mask + -1e9)

    # softmax normalization
    attentionWeights = tf.nn.softmax(logits, axis=-1)

    # final value
    output = tf.matmul(attentionWeights, value)

    return output


class MultiHeadAttention(tf.keras.layers.Layer):

    def __init__(self, d_model, num_heads, name='multi_head_attention'):
        super(MultiHeadAttention, self).__init__(name=name)
        self.d_model = d_model
        self.num_heads = num_heads

        assert self.d_model % self.num_heads == 0
        self.depth = self.d_model // self.num_heads

        self.queryDense = tf.keras.layers.Dense(self.d_model)
        self.keyDense = tf.keras.layers.Dense(self.d_model)
        self.valueDense = tf.keras.layers.Dense(self.d_model)
        self.dense = tf.keras.layers.Dense(self.d_model)

    def splitHeads(self, inputs, batch_size):
        inputs = tf.reshape(inputs, shape=(batch_size, -1, self.num_heads, self.depth))
        return tf.transpose(inputs, perm=[0, 2, 1, 3])

    def call(self, inputs):
        query, key, value, mask = inputs['query'], inputs['key'], inputs['value'], inputs['mask']
        batchSize = tf.shape(query)[0]

        # linear layers
        query = self.queryDense(query)
        key = self.keyDense(key)
        value = self.valueDense(value)

        # split heads
        query = self.splitHeads(query, batchSize)
        key = self.splitHeads(key, batchSize)
        value = self.splitHeads(value, batchSize)

        # scaled dot product attention
        scaledAttention = getScaledDotProductAttention(query, key, value, mask)
        scaledAttention = tf.transpose(scaledAttention, perm=[0, 2, 1, 3])

        # concat heads
        concatAttention = tf.reshape(scaledAttention, (batchSize, -1, self.d_model))

        # final linear layer
        outputs = self.dense(concatAttention)

        return outputs


def createPaddingMask(x):
    mask = tf.cast(tf.math.equal(x, 0), tf.float32)
    # (batch_size, 1, 1, sequence_length)
    return mask[:, tf.newaxis, tf.newaxis, :]


def createLookAheadMask(x):
    seqLen = tf.shape(x)[1]
    lookAheadMask = 1 - tf.linalg.band_part(tf.ones((seqLen, seqLen)), -1, 0)
    paddingMask = createPaddingMask(x)

    return tf.maximum(lookAheadMask, paddingMask)


class PositionalEncoding(tf.keras.layers.Layer):

    def __init__(self, position, d_model):
        super(PositionalEncoding, self).__init__()
        self.posEncoding = self.getPositionalEncoding(position, d_model)

    def getAngles(self, position, i, d_model):
        angles = 1 / tf.pow(10000, (2 * (i // 2)) / tf.cast(d_model, tf.float32))
        return position * angles

    def getPositionalEncoding(self, position, d_model):
        angleRads = self.getAngles(
            position=tf.range(position, dtype=tf.float32)[:, tf.newaxis],
            i=tf.range(d_model, dtype=tf.float32)[tf.newaxis, :],
            d_model=d_model
        )

        # apply sine to even index
        sines = tf.math.sin(angleRads[:, 0::2])
        # apply cosine to odd index
        cosines = tf.math.cos(angleRads[:, 1::2])

        posEncoding = tf.concat([sines, cosines], axis=-1)
        posEncoding = posEncoding[tf.newaxis, ...]

        return tf.cast(posEncoding, tf.float32)

    def call(self, inputs):
        return inputs + self.posEncoding[:, :tf.shape(inputs)[1], :]


def getEncoderLayer(units, d_model, num_heads, dropout, name='encoder_layer'):
    inputs = tf.keras.Input(shape=(None, d_model), name='inputs')
    paddingMask = tf.keras.Input(shape=(1, 1, None), name='padding_mask')

    attention = MultiHeadAttention(
        d_model, num_heads, name='attention')({
            'query': inputs,
            'key': inputs,
            'value': inputs,
            'mask': paddingMask
        })

    attention = tf.keras.layers.Dropout(rate=dropout)(attention)
    attention = tf.keras.layers.LayerNormalization(epsilon=1e-6)(inputs + attention)

    outputs = tf.keras.layers.Dense(units, activation='relu')(attention)
    outputs = tf.keras.layers.Dense(d_model)(outputs)
    outputs = tf.keras.layers.Dropout(dropout)(outputs)
    outputs = tf.keras.layers.LayerNormalization(epsilon=1e-6)(attention + outputs)

    return tf.keras.Model(inputs=[inputs, paddingMask], outputs=outputs, name=name)


def getEncoderBlock(vocab_size, num_layers, units, d_model, num_heads,
                    dropout, name='encoder'):

    inputs = tf.keras.Input(shape=(None,), name='inputs')
    paddingMask = tf.keras.Input(shape=(1, 1, None), name='padding_mask')

    embeddings = tf.keras.layers.Embedding(vocab_size, d_model)(inputs)
    embeddings *= tf.math.sqrt(tf.cast(d_model, tf.float32))
    embeddings = PositionalEncoding(vocab_size, d_model)(embeddings)

    outputs = tf.keras.layers.Dropout(dropout)(embeddings)

    for i in range(num_layers):
        outputs = getEncoderLayer(
            units, d_model, num_heads, dropout,
            name='encoder_layer_{}'.format(i)
        )([outputs, paddingMask])

    return tf.keras.Model(inputs=[inputs, paddingMask], outputs=outputs, name=name)


def getDecoderLayer(units, d_model, num_heads, dropout, name='decoder_layer'):

    inputs = tf.keras.Input(shape=(None, d_model), name='inputs')
    encOutputs = tf.keras.Input(shape=(None, d_model), name='encoder_outputs')
    lookAheadMask = tf.keras.Input(shape=(1, None, None), name='look_ahead_mask')
    paddingMask = tf.keras.Input(shape=(1, 1, None), name='padding_mask')

    attention1 = MultiHeadAttention(
        d_model, num_heads, name='attention_1')(inputs={
            'query': inputs,
            'key': inputs,
            'value': inputs,
            'mask': lookAheadMask
        })

    attention1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)(attention1 + inputs)

    attention2 = MultiHeadAttention(
        d_model, num_heads, name='attention_2')(inputs={
            'query': attention1,
            'key': encOutputs,
            'value': encOutputs,
            'mask': paddingMask
        })

    attention2 = tf.keras.layers.Dropout(dropout)(attention2)
    attention2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)(attention2 + attention1)

    outputs = tf.keras.layers.Dense(units, activation='relu')(attention2)
    outputs = tf.keras.layers.Dense(d_model)(outputs)
    outputs = tf.keras.layers.Dropout(dropout)(outputs)
    outputs = tf.keras.layers.LayerNormalization(epsilon=1e-6)(outputs + attention2)

    return tf.keras.Model(
        inputs=[inputs, encOutputs, lookAheadMask, paddingMask],
        outputs=outputs,
        name=name
    )


def getDecoderBlock(vocab_size, num_layers, units, d_model, num_heads,
                    dropout, name='decoder'):

    inputs = tf.keras.Input(shape=(None,), name='inputs')
    encOutputs = tf.keras.Input(shape=(None, d_model), name='encoder_outputs')
    lookAheadMask = tf.keras.Input(shape=(1, None, None), name='look_ahead_mask')
    paddingMask = tf.keras.Input(shape=(1, 1, None), name='padding_mask')

    embeddings = tf.keras.layers.Embedding(vocab_size, d_model)(inputs)
    embeddings *= tf.math.sqrt(tf.cast(d_model, tf.float32))
    embeddings = PositionalEncoding(vocab_size, d_model)(embeddings)

    outputs = tf.keras.layers.Dropout(dropout)(embeddings)

    for i in range(num_layers):
        outputs = getDecoderLayer(
            units, d_model, num_heads, dropout,
            name='decoder_layer_{}'.format(i)
        )(inputs=[outputs, encOutputs, lookAheadMask, paddingMask])

    return tf.keras.Model(
        inputs=[inputs, encOutputs, lookAheadMask, paddingMask],
        outputs=outputs,
        name=name
    )


def getTransformerModel(vocab_size, num_layers, units, d_model, num_heads,
                        dropout, name='transformer'):

    inputs = tf.keras.Input(shape=(None,), name='inputs')
    decInputs = tf.keras.Input(shape=(None,), name='dec_inputs')

    encPaddingMask = tf.keras.layers.Lambda(
        createPaddingMask, output_shape=(1, 1, None),
        name='enc_padding_mask')(inputs)

    # mask the future tokens for decoder inputs at 1st attention block
    lookAheadMask = tf.keras.layers.Lambda(
        createLookAheadMask, output_shape=(1, None, None),
        name='look_ahead_mask')(decInputs)

    # mask the encoder inputs at the 2nd attention block
    decPaddingMask = tf.keras.layers.Lambda(
        createPaddingMask, output_shape=(1, 1, None),
        name='dec_padding_mask')(inputs)

    encOutputs = getEncoderBlock(
        vocab_size, num_layers, units, d_model, num_heads, dropout
    )([inputs, encPaddingMask])

    decOutputs = getDecoderBlock(
        vocab_size, num_layers, units, d_model, num_heads, dropout
    )(inputs=[decInputs, encOutputs, lookAheadMask, decPaddingMask])

    outputs = tf.keras.layers.Dense(vocab_size, name='outputs')(decOutputs)

    return tf.keras.Model(inputs=[inputs, decInputs], outputs=outputs, name=name)


def customLossFunction(y_true, y_pred):
    y_true = tf.reshape(y_true, shape=(-1, maxlen - 1))

    loss = tf.keras.losses.SparseCategoricalCrossentropy(
        from_logits=True, reduction='none')(y_true, y_pred)

    mask = tf.cast(tf.not_equal(y_true, 0), tf.float32)
    loss = tf.multiply(loss, mask)

    return tf.reduce_mean(loss)


class CustomSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):

    def __init__(self, d_model, warmup_steps=4000):
        super(CustomSchedule, self).__init__()

        self.d_model = d_model
        self.d_model = tf.cast(self.d_model, tf.float32)
        self.warmup_steps = warmup_steps

    def __call__(self, step):
        arg1 = tf.math.rsqrt(step)
        arg2 = step * (self.warmup_steps**-1.5)

        return tf.math.rsqrt(self.d_model) * tf.math.minimum(arg1, arg2)


def customAccuracyMetric(y_true, y_pred):
    y_true = tf.reshape(y_true, shape=(-1, maxlen - 1))
    return tf.keras.metrics.sparse_categorical_accuracy(y_true, y_pred)
