import gym
import numpy as np
import matplotlib.pyplot as plt
from ppo.buffer_of_samples import BufferOfSamples
from ppo.training_graph import TrainingGraph


def plot(*means_stds_name, title="", xlabel="", ylabel=""):
    """Generate a colorful plot with the provided data."""
    plt.figure()
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    for indices, means, stds, name in means_stds_name:
        ccolor = np.random.rand(3,)
        plt.fill_between(indices, means - stds, means + stds, color=np.hstack([ccolor, [0.2]]))
        plt.plot(indices,means, color=np.hstack([ccolor, [1.0]]), label=name)
    plt.legend(loc=4)
    plt.savefig(title + ".png")
    plt.close()


if __name__ == "__main__":

    env = gym.make("CartPole-v0")
    env.seed(0)

    agent = TrainingGraph(16, 2, True, 0.1, 
            .9, 1.0, 10, 4,
            2,
            0.2,
            4,
            0.0005)

    buffer = BufferOfSamples(11, 4, 10)

    logged_trials = []
    for t in range(1):

        logged_rewards = []
        agent.reset()
        for i in range(10):

            while not buffer.is_full():

                state = env.reset()
                done = False
                while not done and not buffer.is_full():

                    action, probability = agent.act([[state]])
                    action, probability = action[0, 0], probability[0, 0]

                    next_state, reward, done, _info = env.step(action)

                    buffer.add(state, action, probability, reward, next_state)
                    state = next_state

                buffer.finished_episode()

            samples = buffer.sample()
            states, actions, probabilities, rewards, next_states = [], [], [], [], []

            for sample in samples:

                states.append([])
                actions.append([])
                probabilities.append([])
                rewards.append([])
                next_states.append([])

                for timestep in sample:
                    states[-1].append(timestep[0])
                    actions[-1].append(timestep[1])
                    probabilities[-1].append(timestep[2])
                    rewards[-1].append(timestep[3])
                    next_states[-1].append(timestep[4])

            states = np.array(states)
            actions = np.array(actions)
            probabilities = np.array(probabilities)
            rewards = np.array(rewards)
            next_states = np.array(next_states)

            agent.step(states, actions, probabilities, rewards, next_states)
            buffer.empty()

            logged_rewards.append(np.mean(np.sum(rewards, axis=1), axis=0))
            print("On training step {0} average utility was {1}.".format(i, logged_rewards[-1]))

        logged_trials.append(logged_rewards)

    env.close()
    trajectories = np.array(logged_trials)

    plot(
        (np.arange(10), np.mean(trajectories, axis=0), 
            np.std(trajectories, axis=0), "REINFORCE Policy Gradient"),
        title="Training A REINFORCE Policy On {0}".format("CartPole-v0"), 
        xlabel="Iteration", 
        ylabel="Expected Future Reward")