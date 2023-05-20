import gymnasium as gym
from PolicyAgent import DQNagent
from collections import namedtuple
import torch


EPISODES = 100
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
env = gym.make('CartPole-v1', render_mode="human")
agent = DQNagent(env, 4, 2, torch.nn.HuberLoss(), device)

for n in range(EPISODES):
    steps = 0
    state, _ = env.reset()
    state = torch.tensor(state, dtype=torch.float32,
                         device=device).unsqueeze(0)
    previousReward = None

    while True:
        action = env.action_space.sample()
        action = agent.step(state, previousReward, steps=steps)
        observation, reward, terminated, truncated, _ = env.step(action.item())
        observation = torch.tensor(
            observation, dtype=torch.float32, device=device).unsqueeze(0)  # voglio che abbia già implementata la dimensione dei batch
        reward = torch.tensor(reward, dtype=torch.float32,
                              device=device).unsqueeze(0)
        previousReward = reward

        state = observation
        agent.optimize()

        if terminated or truncated:
            break

        steps += 1

env.close()
