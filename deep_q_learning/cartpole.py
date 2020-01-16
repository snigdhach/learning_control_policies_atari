import gym
from neural_network import neuralNetwork
import numpy as np
import os
import sys

sys.path.append("benchmark/")

from benchmark import benchmark

# Parameters and toggles
ENV_NAME = "CartPole-v0"
NOS_EPISODES = 250
SAVE_PATH = "./models/"
if not os.path.exists(SAVE_PATH):
    os.mkdir(SAVE_PATH)
RENDER_GAME = False

# Setting up Gym
env = gym.make(ENV_NAME)
logger = benchmark()

# Finding the Network Shape
action_space = env.action_space.n
observation_space = env.observation_space.shape

# Initializing neural network
nn_solver = neuralNetwork(action_space, observation_space)

# Running Q Learning
for episode in range(NOS_EPISODES):
    state = env.reset()
    state = np.reshape(state, (1, observation_space[0]))
    time_step = 0
    terminate = False
    while True:
        # Render Game
        if RENDER_GAME:
            env.render()
        # Increment Time Step
        time_step = time_step + 1
        # Take Action
        action = nn_solver.act(state, env)
        next_state, reward, terminate, info = env.step(action)
        # Save the data generated for training
        next_state = np.reshape(next_state, (1, observation_space[0]))
        if terminate:
            # To penalize bad action
            reward = -reward
        nn_solver.remember(state, action, reward, next_state, terminate)
        # Update State
        state = next_state
        # End of Epsiode
        if terminate:
            msg = "Episode: {}, Epsilon: {}, Step: {} \n"\
                .format((episode+1), nn_solver.epsilon, time_step)
            logger.record_score(episode+1, time_step)
            print (msg)
            break
        nn_solver.learn()
nn_solver.save(SAVE_PATH)
env.close()
logger.display_log()
logger.stats()
logger.plot_log()