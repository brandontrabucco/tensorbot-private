import tensorflow as tf

class PPO(tf.keras.layers.Layer):

    def __init__(self, epsilon):
        self.epsilon = epsilon
    
    def __call__(self, sampling_distribution_probabilities, 
                       policy_distribution_probabilities, 
                       advantages):
        """Args: sampling_distribution_probabilities: a float32 Tensor of shape [num_samples, trajectory_size]
                    policy_distribution_probabilities: a float32 Tensor of shape [num_samples, trajectory_size]
                    advantages: a float32 Tensor of shape [num_samples, trajectory_size]
        Returns: a float32 Scalar that optimizes the policy."""
        probability_ratios = policy_distribution_probabilities / sampling_distribution_probabilities
        clipped_reward = tf.minimum(
            probability_ratios * advantages, 
            tf.clip_by_value(
                probability_ratios, 
                1 - self.epsilon, 
                1 + self.epsilon) * advantages)
        return tf.reduce_mean(clipped_reward)