import tensorflow as tf
from ppo.stem import Stem
from ppo.value_function import ValueFunction
from ppo.gae import GAE
from ppo.policy import Policy
from ppo.ppo import PPO

class TrainingGraph(object):

    def __init__(self, 
            hidden_size, num_layers, use_residual, dropout_rate, 
            discount_factor, lambda_factor, num_samples, trajectory_size,
            num_actions,
            epsilon,
            state_cardinality,
            learning_rate):
        self.graph = tf.Graph()
        with self.graph.as_default():
            self.stem = Stem(hidden_size, num_layers, use_residual, dropout_rate)
            self.value_function = ValueFunction()
            self.gae = GAE(discount_factor, lambda_factor, num_samples, trajectory_size)
            self.policy = Policy(num_actions)
            self.ppo = PPO(epsilon)
            self.state_feed = tf.placeholder(tf.float32, shape=[None, None, state_cardinality])
            self.action_feed = tf.placeholder(tf.int32, shape=[None, None])
            self.probability_feed = tf.placeholder(tf.float32, shape=[None, None])
            self.reward_feed = tf.placeholder(tf.float32, shape=[None, None])
            self.next_state_feed = tf.placeholder(tf.float32, shape=[None, None, state_cardinality])
            state_embeddings = self.stem(self.state_feed)
            values = self.value_function(state_embeddings)
            next_state_embeddings = self.stem(self.next_state_feed)
            next_values = self.value_function(next_state_embeddings)
            advantages, value_loss = self.gae(values, next_values, self.reward_feed)
            self.sampled_actions, self.sampled_probabilities, _a = self.policy(state_embeddings)
            _b, action_probabilities, entropy = self.policy(state_embeddings, actions=self.action_feed)
            ppo_reward = self.ppo(self.probability_feed, action_probabilities, advantages)
            self.objective_function = value_loss - ppo_reward - entropy
            self.learning_step = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(
                self.objective_function, var_list=(
                    self.stem.trainable_variables + 
                    self.value_function.trainable_variables + 
                    self.policy.trainable_variables))
            self.initializer = tf.global_variables_initializer()
            self.sess = tf.Session(graph=self.graph)
        
    def reset(self):
        self.sess.run(self.initializer)

    def save(self):
        pass

    def load(self):
        pass

    def step(self, batch_of_states,
            batch_of_actions, batch_of_probabilities,
            batch_of_rewards, batch_of_next_states ):
        self.sess.run(self.learning_step, feed_dict={
            self.state_feed: batch_of_states,
            self.action_feed: batch_of_actions,
            self.probability_feed: batch_of_probabilities,
            self.reward_feed: batch_of_rewards,
            self.next_state_feed: batch_of_next_states})

    def act(self, batch_of_states):
        return self.sess.run([self.sampled_actions, self.sampled_probabilities], feed_dict={
            self.state_feed: batch_of_states})