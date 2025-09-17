import gymnasium as gym
import numpy as np
import random
from IPython.display import clear_output
import time
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation # Para criar animações

env = gym.make('FrozenLake-v1', is_slippery=False, render_mode='rgb_array')

n_states = env.observation_space.n
n_actions = env.action_space.n

q_table = np.zeros((n_states, n_actions))

learning_rate_alpha = 0.9
discount_factor_gamma = 0.9
epsilon = 1.0

max_epsilon = 1.0
min_epsilon = 0.01 
decay_rate = 0.001  

total_eps = 2000

print(q_table)

rewards_per_episode = []

for i, episode in enumerate(range(total_eps)):
    state, info = env.reset()
    done = False
    truncated = False
    current_episode_reward = 0

    while not done and not truncated:
        if random.uniform(0, 1) < epsilon:
            action = env.action_space.sample()
        else:
            action = np.argmax(q_table[state, :])
        new_state, reward, done, truncated, info = env.step(action)
        old_q_value = q_table[state, action]
        next_max_q_value = np.max(q_table[new_state, :])
        new_q_value = old_q_value + learning_rate_alpha * \
            (reward + discount_factor_gamma * next_max_q_value - old_q_value)
        
        q_table[state, action] = new_q_value
        state = new_state
        current_episode_reward += reward
    
    epsilon = min_epsilon + (max_epsilon - min_epsilon) * np.exp(-decay_rate * episode)
    rewards_per_episode.append(current_episode_reward)

    if episode % 100 == 0:
        print(f"Episódio: {episode}/{total_eps}, Epsilon: {epsilon:.2f}, Recompensa Média (últimos 100): {np.mean(rewards_per_episode[-100:]):.2f}")
        time.sleep(0.5)

num_test_episodes = 3
frames = [] 

for episode in range(num_test_episodes):
    state, info = env.reset()
    done = False
    truncated = False

    frames.append(env.render())

    while not done and not truncated:
        action = np.argmax(q_table[state, :])
        new_state, reward, done, truncated, info = env.step(action)
        frames.append(env.render())
    
env.close()

fig, ax = plt.subplots()
img = ax.imshow(frames[0])
ax.axis('off')

def update(frame):
    img.set_array(frame)
    return [img]

ani = FuncAnimation(fig, update, frames=frames, blit=True, interval=100) 

plt.show() 

plt.close(fig)