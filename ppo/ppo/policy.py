import tensorflow as tf

def collapse_dims(tensor, flat_points):
    flat_size = tf.shape(tensor)[flat_points[0]]
    for i in flat_points[1:]:
        flat_size = flat_size * tf.shape(tensor)[i]
    fixed_points = [i for i in range(len(tensor.shape)) if i not in flat_points]
    fixed_shape = [tf.shape(tensor)[i] for i in fixed_points]
    tensor = tf.transpose(tensor, fixed_points + flat_points)
    final_points = list(range(len(fixed_shape)))
    final_points.insert(flat_points[0], len(fixed_shape))
    return tf.transpose(tf.reshape(tensor, fixed_shape + [flat_size]), final_points)

def tile_with_new_axis(tensor, repeats, locations):
    repeats, locations = zip(*sorted(zip(repeats, locations), key=lambda ab: ab[1]))
    for i in sorted(locations):
        tensor = tf.expand_dims(tensor, i)
    reverse_d = {val: idx for idx, val in enumerate(locations)}
    tiles = [repeats[reverse_d[i]] if i in locations else 1 for i, _s in enumerate(tensor.shape)]
    return tf.tile(tensor, tiles)

class Policy(tf.keras.layers.Layer):

    def __init__(self, num_actions, **kwargs):
        super(Policy, self).__init__(**kwargs)
        self.num_actions = num_actions
        self.policy_layer = tf.layers.Dense(self.num_actions, 
            kernel_initializer=tf.contrib.layers.xavier_initializer())

    def __call__(self, states, actions=None):
        logits = self.policy_layer(states)
        distribution = tf.nn.softmax(logits)
        num_samples = tf.shape(logits)[0]
        trajectory_size = tf.shape(logits)[1]
        if actions is None:
            logits = tf.reshape(logits, [num_samples * trajectory_size, self.num_actions])
            actions = tf.squeeze(tf.multinomial(logits, 1, output_dtype=tf.int32), axis=1)
            actions = tf.reshape(actions, [num_samples, trajectory_size])
        sample_ids = tile_with_new_axis(tf.range(num_samples), [trajectory_size], [1])
        trajectory_ids = tile_with_new_axis(tf.range(trajectory_size), [num_samples], [0])
        probabilities = tf.gather_nd(distribution, tf.stack([sample_ids, trajectory_ids, actions], axis=2))
        entropy = -tf.reduce_mean(tf.reduce_sum(distribution * tf.log(distribution), axis=2))
        return actions, probabilities, entropy

    @property
    def trainable_variables(self):
        policy_variables = self.policy_layer.trainable_variables
        return policy_variables

    @property
    def trainable_weights(self):
        return self.trainable_variables

    @property
    def variables(self):
        policy_variables = self.policy_layer.variables
        return policy_variables

    @property
    def weights(self):
        return self.variables