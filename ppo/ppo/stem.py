import tensorflow as tf

class Stem(tf.keras.layers.Layer):

    def __init__(self, hidden_size, num_layers, use_residual, dropout_rate, **kwargs):
        super(Stem, self).__init__(**kwargs)
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.use_residual = use_residual
        self.dropout_rate = dropout_rate
        self.dropout_layers = [tf.layers.Dropout(
            rate=self.dropout_rate) for i in range(self.num_layers)]
        self.hidden_layers = [tf.layers.Dense(
            self.hidden_size, kernel_initializer=tf.contrib.layers.xavier_initializer(),
            activation=tf.nn.relu) for i in range(self.num_layers)]

    def __call__(self, inputs):
        x = self.dropout_layers[0](self.hidden_layers[0](inputs))
        for dropout, layer in zip(self.dropout_layers[1:], self.hidden_layers[1:]):
            x = dropout(layer(x)) + (x if self.use_residual else 0.)
        return x

    @property
    def trainable_variables(self):
        stem_variables = [p.trainable_variables for p in self.hidden_layers]
        return stem_variables

    @property
    def trainable_weights(self):
        return self.trainable_variables

    @property
    def variables(self):
        stem_variables = [p.variables for p in self.hidden_layers]
        return stem_variables

    @property
    def weights(self):
        return self.variables