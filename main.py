import gymnasium as gym

EPISODES = 1
env = gym.make('CartPole-v1', render_mode="human")

for n in range(EPISODES):
    steps = 0
    state, _ = env.reset()

    while True:
        observation, reward, terminated, truncated, _ = env.step(
            env.action_space.sample())

        if terminated or truncated:
            break

        steps += 1

env.close()
