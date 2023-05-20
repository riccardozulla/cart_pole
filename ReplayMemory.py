import torch
from collections import deque, namedtuple
import random

Transition = namedtuple(
    'Transition', ('state', 'action', 'next_state', 'reward'))


class ReplayMemory():
    def __init__(self, capacity=1000) -> None:
        self.memory = deque([], capacity)

    def push(self, *args):
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)
