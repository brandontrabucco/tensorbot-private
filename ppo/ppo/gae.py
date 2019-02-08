import tensorflow as tf

class GAE(tf.keras.layers.Layer):

    def __init__(self, discount_factor, lambda_factor, num_samples, trajectory_size):
        self.discount_factor = discount_factor
        self.lambda_factor = lambda_factor
        self.reward_coefficients = tf.cumprod(tf.fill([num_samples, trajectory_size], 
            discount_factor), axis=1, exclusive=True)
        self.delta_t_coefficients = tf.cumprod(tf.fill([num_samples, trajectory_size], 
            discount_factor * lambda_factor), axis=1, exclusive=True)
    
    def __call__(self, values, 
                       next_values, 
                       rewards):
        """Args: values: a float32 Tensor of shape [num_samples, trajectory_size]
                    next_values: a float32 Tensor of shape [num_samples, trajectory_size]
                    rewards: a float32 Tensor of shape [num_samples, trajectory_size]
        Returns: a float32 Scalar loss function for Value estimation.
                    a float32 estimate of the Advantage function with shape [num_samples, trajectory_size]."""
        weighted_rewards = rewards * self.reward_coefficients
        weighted_target_values = tf.cumsum(weighted_rewards, axis=1, reverse=True)
        target_values = weighted_target_values / self.reward_coefficients
        value_function_losses = tf.nn.l2_loss(values - target_values)
        delta_ts = rewards + self.discount_factor * next_values - values
        weighted_delta_ts = delta_ts * self.delta_t_coefficients
        weighted_advantages = tf.cumsum(weighted_delta_ts, axis=1, reverse=True)
        advantages = weighted_advantages / self.delta_t_coefficients
        return tf.stop_gradient(advantages), tf.reduce_mean(value_function_losses)