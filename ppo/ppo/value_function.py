import tensorflow as tf

class ValueFunction(tf.keras.layers.Layer):

    def __init__(self, **kwargs):
        super(ValueFunction, self).__init__(**kwargs)
        self.value_layer = tf.layers.Dense(1, 
            kernel_initializer=tf.contrib.layers.xavier_initializer()) 

    def __call__(self, inputs):
        return tf.squeeze(self.value_layer(inputs), axis=-1)

    @property
    def trainable_variables(self):
        value_variables = self.value_layer.trainable_variables
        return value_variables

    @property
    def trainable_weights(self):
        return self.trainable_variables

    @property
    def variables(self):
        value_variables = self.value_layer.variables
        return value_variables

    @property
    def weights(self):
        return self.variables