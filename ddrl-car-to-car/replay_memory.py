from collections import deque
import random


class ReplayMemory:

    def __init__(self, max_len=1, min_len=1):
        self.memory = deque(maxlen=max_len)
        self.min_len = min_len

    def add_transition(self, transition):
        self.memory.append(transition)

    def has_enough_values(self):
        return len(self.memory) > self.min_len

    def get_random_samples(self, batch_size):
        return random.sample(self.memory, batch_size) if len(self.memory) > batch_size else []


class Transition:

    def __init__(self, old_state, new_state, action, reward, is_done=False):
        self.old_state = old_state
        self.new_state = new_state
        self.action = action
        self.reward = reward
        self.is_done = is_done
