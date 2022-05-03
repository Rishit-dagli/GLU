import tensorflow as tf


class GLU(tf.keras.layers.Layer):
    def __init__(self, size, bias=True, **kwargs):
        super(GLU, self).__init__(**kwargs)
        self.size = size
        self.dense1 = tf.keras.layers.Dense(size, use_bias=bias)
        self.dense2 = tf.keras.layers.Dense(size, use_bias=bias)

    def call(self, x):
        chunk1 = self.dense1(x)
        chunk2 = tf.sigmoid(x) * self.dense2(x)
        x = tf.multiply(chunk1, chunk2)
        return x
