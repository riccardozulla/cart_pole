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
    # Take 100 episode averages and plot them too

    # if len(durations_t) >= 100:
    #     means = durations_t.unfold(0, 100, 1).mean(1).view(-1)
    #     means = torch.cat((torch.zeros(99), means))
    #     plt.plot(means.numpy())

    cumulative_sum = torch.cumsum(durations_t, dim=0)
    means = cumulative_sum / torch.arange(1, len(durations_t) + 1)
    plt.plot(means.numpy())

    plt.pause(0.001)  # pause a bit so that plots are updated


EPISODES = 600 #600
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
env = gym.make('CartPole-v1') #, render_mode = "human")
# agent = DQNagent(env, 4, 2, torch.nn.HuberLoss(), device)
agent = DQNagent(env, 4, 2, torch.nn.SmoothL1Loss(), device) 

episode_durations = []

for n in range(EPISODES):
    steps = 0
    state, _ = env.reset()
    state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
    # print("Initial: ", state)
    previousReward = None
    # print(previousReward)
    # total_reward = 0
    # while True:
    for t in count():
        action = env.action_space.sample()
        action = agent.step(state, previousReward)
        state, reward, terminated, truncated, _ = env.step(action.item())
        state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
        previousReward = torch.tensor([reward], device=device) # le quadre sono fondamentali
        # done = terminated or truncated
        if truncated: # succeded!
            print("Succeded!")
            action = agent.step(state, torch.tensor([10], device=device))
            episode_durations.append(t + 1)
            plot_durations()
            break
        elif terminated: # failed
            print("Failed.")
            action = agent.step(state, torch.tensor([0], device=device))
            episode_durations.append(t + 1)
            plot_durations()
            break
    # print(reward)

print('Complete')
plot_durations(show_result=True)
env.close()

plt.ioff()
plt.show()

