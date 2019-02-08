import random

class BufferOfSamples(object):

    def __init__(self, max_num_trajectories, max_trajectory_length, batch_size):
        """Implements a buffer of samples from the Environment."""
        self.max_num_trajectories = max_num_trajectories
        self.max_trajectory_length = max_trajectory_length
        self.batch_size = batch_size
        self.empty()

    def empty(self):
        """Clears the current buffer of samples."""
        self.trajectories = []
        self.finished_episode()

    def num_trajectories(self):
        """Returns: the total length of the buffer."""
        return len(self.trajectories)

    def fill_ratio(self):
        """Returns: the fraction of the buffer that is used."""
        return self.num_trajectories() / self.max_num_trajectories

    def is_full(self):
        """Returns: whether the buffer is at capacity."""
        return self.num_trajectories() >= self.max_num_trajectories

    def is_complete(self):
        """Returns: whether the last trajectory has been fully collected."""
        return len(self.working_trajectory) >= self.max_trajectory_length

    def sample(self):
        """Returns: a list of trajectories."""
        return random.sample(self.trajectories, self.batch_size)

    def add(self, state, action, probability, reward, next_state):
        """Args: state, a float32 tensor shape [state_size].
                 action, a int32 scalar
                 probability, a float32 scalar
                 reward, a float32 scalar
                 next_state, a float32 tensor shape [state_size].
        Returns: None."""
        if self.is_complete():
            if self.is_full():
                self.trajectories = self.trajectories[1:]
            self.trajectories = self.trajectories + [self.working_trajectory]
            self.finished_episode()
        self.working_trajectory.append([state, action, probability, reward, next_state])
        
    def finished_episode(self):
        """Flags that an episode of samples has finished being collected."""
        self.working_trajectory = []