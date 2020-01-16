"""
Open AI Gym API Sample
"""

#Import Library
import gym

#Set Environment
env = gym.make('CartPole-v0')
env.reset()

#Steps for N interations
N = 1000
for _ in range(N):
    env.render()    #Render Game
    env.step(env.action_space.sample()) #Take a random action
