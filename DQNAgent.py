import torch.nn as nn
import torch.nn.functional as F
from collections import deque, namedtuple
import torch
import random
import math

Transition = namedtuple(
    'Transition', ('state', 'action', 'reward', 'next_state'))


class DQNagent():
    def __init__(self, env, nInputs, nOutputs, criterion, device) -> None:

        self.BATCH_SIZE = 128
        self.GAMMA = 0.99
        self.EPS_START = 0.9
        self.EPS_END = 0.05
        self.EPS_DECAY = 1000
        self.TAU = 0.005
        self.LR = 1e-3

        self.env = env
        self.device = device

        self.previousState = None

        self.policy_net = DQN(nInputs, nOutputs).to(device)
        self.target_net = DQN(nInputs, nOutputs).to(device)

        # copia la policy e la mette nella target.
        self.target_net.load_state_dict(self.policy_net.state_dict())

        self.criterion = criterion
        self.optimizer = torch.optim.AdamW(
            self.policy_net.parameters(), lr=self.LR)

        self.memory = ReplayMemory(10000)

        self.previousState = None
        self.previousAction = None

    def step(self, state, previousReward, steps):
        if (previousReward is not None):
            # salva in memoria
            self.memory.push(self.previousState,
                             self.previousAction, previousReward, state)

        ESP_THRESHOLD = self.EPS_END + \
            (self.EPS_START - self.EPS_END) * \
            math.exp(-1. * steps / self.EPS_DECAY)

        sample = random.random()

        if sample > ESP_THRESHOLD:
            action = torch.tensor(
                [self.env.action_space.sample()], dtype=torch.long, device=self.device)
        else:
            action = self.policy_net(state).max(1)[1]

        self.previousState = state
        self.previousAction = action
        return action

    def optimize(self):
        if len(self.memory) < self.BATCH_SIZE:
            return

        transitions = self.memory.sample(self.BATCH_SIZE)

        batch = Transition(*zip(*transitions))
        # batch.state mi passa un array di tutti gli stati
        state = torch.cat(batch.state)
        reward = torch.cat(batch.reward)  # prendi tutti i rewards
        action = torch.cat(batch.action)  # prendi tutte le actions
        next_state = torch.cat(batch.next_state)  # ...

        with torch.no_grad():
            estimated_Qs = reward + self.GAMMA * \
                self.target_net(next_state).max(1)[0]
        predicted_Qs = self.policy_net(state).gather(
            1, action.unsqueeze(0)).squeeze(0)
        loss = self.criterion(predicted_Qs, estimated_Qs)
        self.optimizer.zero_grad()
        loss.backward()

        self.optimizer.step()

        target_weights = self.target_net.state_dict()  # prende i weights
        policy_weights = self.policy_net.state_dict()

        # aggiorna i pesi della target di TAU
        for key in target_weights.keys():
            target_weights[key] = \
                (1-self.TAU) * target_weights[key] + \
                self.TAU * policy_weights[key]

        self.target_net.load_state_dict(target_weights)


class ReplayMemory():
    def __init__(self, capacity=1000) -> None:
        self.memory = deque([], capacity)

    def push(self, *args):
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


class DQN(nn.Module):

    def __init__(self, n_observations, n_actions):
        # inizializza module, trovare commento adeguato
        super(DQN, self).__init__()
        self.layer1 = nn.Linear(n_observations, 128)
        self.layer2 = nn.Linear(128, 128)
        self.layer3 = nn.Linear(128, n_actions)

    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        return self.layer3(x)
