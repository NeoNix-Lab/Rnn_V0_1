import random
from collections import deque
import numpy as np

class ReplayBuffer:
    #TODO:verificare il funzionamento di questo buffer
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)
        self.capacity = capacity

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        if len(self.buffer) < batch_size:
            return None

        batch = random.sample(self.buffer, min(len(self.buffer), batch_size))
        state_batch = np.stack([x[0] for x in batch])
        action_batch = np.array([x[1] for x in batch])
        reward_batch = np.array([x[2] for x in batch])
        next_state_batch = np.stack([x[3] for x in batch])
        done_batch = np.array([x[4] for x in batch])

        return state_batch, action_batch, reward_batch, next_state_batch, done_batch

    def __len__(self):
        return len(self.buffer)

    def clear(self):
        self.buffer.clear
