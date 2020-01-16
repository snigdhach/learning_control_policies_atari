"""
Q Learning on Atari Tennis
Author: Harsh Bhate
Date: March 25
Rev: 1.0 
"""

# Import Library
import gym
import numpy as np
import random

# Set Environment
env = gym.make('Tennis-v0')
env.reset()

# Getting Environment Statistics
print ("Action Space {}".format(env.action_space))
print ("State Space {}".format(env.observation_space))

# Setting up Q-Table
H,W,C = np.shape(env.observation_space)
S = H*W*C
q_table = np.zeros([S, env.action_space.n])

# Hyperparameters
alpha = 0.1
gamma = 0.6
epsilon = 0.1

# Plotting Metrics
all_epochs = []
all_penalties = []

# Training the Agent
N = 100001
for i in range(1,N):
    observation = env.reset()
    epochs, penalties, reward = 0, 0, 0
    done = False
    while not done:
        if random.uniform(0, 1) < epsilon:
            action = env.action_space.sample() # Explore action space
        else:
            action = np.argmax(q_table[state]) # Exploit learned values

        next_state, reward, done, info = env.step(action) 
        
        old_value = q_table[state, action]
        next_max = np.max(q_table[next_state])
        
        new_value = (1 - alpha) * old_value + alpha * (reward + gamma * next_max)
        q_table[state, action] = new_value

        if reward == -10:
            penalties += 1

        state = next_state
        epochs += 1
        
    if i % 100 == 0:
        clear_output(wait=True)
        print(f"Episode: {i}")
print ("Training Finished. \n")

# Evaluating the Agent
"""Evaluate agent's performance after Q-learning"""

total_epochs, total_penalties = 0, 0
episodes = 100

for _ in range(episodes):
    state = env.reset()
    epochs, penalties, reward = 0, 0, 0
    
    done = False
    
    while not done:
        action = np.argmax(q_table[state])
        state, reward, done, info = env.step(action)
        print (state)
        if reward == -10:
            penalties += 1

        epochs += 1

    total_penalties += penalties
    total_epochs += epochs

print(f"Results after {episodes} episodes:")
print(f"Average timesteps per episode: {total_epochs / episodes}")
print(f"Average penalties per episode: {total_penalties / episodes}")


env.close()