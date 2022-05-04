import tensorflow as tf



class GLU(tf.keras.layers.Layer):
    def __init__(self, bias = True, dim=-1, **kwargs):
        super(GLU, self).__init__(**kwargs)
        self.bias = bias
        self.dim = dim
        self.dense = tf.keras.layers.Dense(2, use_bias=bias)
    
    def call(self, x):
        out, gate = tf.split(x, 2, axis=self.dim)
        gate =  tf.sigmoid(gate)
        x = tf.multiply(out, gate)
        return x