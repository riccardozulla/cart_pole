import gymnasium as gym
from DQNAgent import DQNagent
import torch
import matplotlib.pyplot as plt
from itertools import count

def plot_durations(show_result=False):
    plt.figure(1)
    durations_t = torch.tensor(episode_durations, dtype=torch.float)
    if show_result:
        plt.title('Result')
    else:
        plt.clf()
        plt.title('Training...')
    plt.xlabel('Episode')
    plt.ylabel('Duration')
    plt.plot(durations_t.numpy())
    plt.pause(0.001)


EPISODES = 600
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
env = gym.make('CartPole-v1') #, render_mode = "human")
# agent = DQNagent(env, 4, 2, torch.nn.HuberLoss(), device)
# agent = DQNagent(4, 2, torch.nn.SmoothL1Loss(), device) 
agent = DQNagent(4, 2, torch.nn.SmoothL1Loss(), device) 


episode_durations = []

for n in range(EPISODES):
    #steps = 0
    state, _ = env.reset()
    state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
    previousReward = None
    for t in count():
        action = agent.step(state, previousReward)
        state, reward, terminated, truncated, _ = env.step(action.item())
        state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
        previousReward = torch.tensor([reward], device=device) # le quadre sono fondamentali
        if truncated: # succeded!
            print("Succeded!")
            action = agent.step(state, torch.tensor([10], device=device))
            episode_durations.append(t + 1)
            plot_durations()
            agent.epsDecay()
            break
        elif terminated: # failed
            print("Failed. Duration: ", t)
            action = agent.step(state, torch.tensor([-1], device=device))
            episode_durations.append(t + 1)
            plot_durations()
            break
print('Complete')
env.close()

plt.ioff()
plt.show()

