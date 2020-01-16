 # -*- coding: utf-8 -*-
"""
Created on Wed Apr 17 02:02:35 2019

@author: User
"""

import sys
import gym
import pylab
import random
import numpy as np
from collections import deque
from keras.layers import Dense
from keras.optimizers import Adam
from keras.models import Sequential


class neuralnet:

    def __init__(self, state_size, action_size):
        self.render = True
        self.load_model = False
        verbose = False
        learning_rate = 0.001
        BATCH_SIZE = 64
        EPSILON_MAX = 1.0
        EPSILON_MIN = 0.01
        EPSILON_DECAY = 0.999
        GAMMA = 0.99
        MEMORY_SIZE = 2000
        self.state_size = state_size
        self.action_size = action_size
        self.discount_factor = GAMMA
        self.learning_rate = learning_rate
        self.epsilon = EPSILON_MAX
        self.epsilon_decay = EPSILON_DECAY
        self.epsilon_min = EPSILON_MIN
        self.batch_size = BATCH_SIZE
        self.train_start = 1000
        self.memory = deque(maxlen=MEMORY_SIZE)

        self.model = Sequential()
        self.model.add(Dense(24, input_dim=self.state_size, activation='relu',
                        kernel_initializer='he_uniform'))
        self.model.add(Dense(24, activation='relu',
                        kernel_initializer='he_uniform'))
        self.model.add(Dense(self.action_size, activation='linear',
                        kernel_initializer='he_uniform'))
        self.model.summary()
        self.model.compile(loss='mse', optimizer=Adam(lr=self.learning_rate))

        self.target_model = Sequential()
        self.target_model.add(Dense(24, input_dim=self.state_size, activation='relu',
                        kernel_initializer='he_uniform'))
        self.target_model.add(Dense(24, activation='relu',
                        kernel_initializer='he_uniform'))
        self.target_model.add(Dense(self.action_size, activation='linear',
                        kernel_initializer='he_uniform'))
        self.target_model.summary()
        self.target_model.compile(loss='mse', optimizer=Adam(lr=self.learning_rate))

        self.update_model()


    def epsilon_greedy(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        else:
            qval = self.model.predict(state)
            return np.argmax(qval[0])


    
    def learn(self):
        if len(self.memory) < self.train_start:
            return
        batch_size = min(self.batch_size, len(self.memory))
        mini_batch = random.sample(self.memory, batch_size)

        update_input = np.zeros((batch_size, self.state_size))
        updatedgoal = np.zeros((batch_size, self.state_size))
        action, reward, done = [], [], []

        for i in range(batch_size):
            update_input[i] = mini_batch[i][0]
            action.append(mini_batch[i][1])
            reward.append(mini_batch[i][2])
            updatedgoal[i] = mini_batch[i][3]
            done.append(mini_batch[i][4])

        target = self.model.predict(update_input)
        target_next = self.model.predict(updatedgoal)
        target_val = self.target_model.predict(updatedgoal)

        for i in range(self.batch_size):
            if done[i]:
                target[i][action[i]] = reward[i]
            else:
                a = np.argmax(target_next[i])
                target[i][action[i]] = reward[i] + self.discount_factor * (
                    target_val[i][a])

        self.model.fit(update_input, target, batch_size=self.batch_size,
                       epochs=1, verbose=0)




    def sampleadd(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay  


    def update_model(self):
        self.target_model.set_weights(self.model.get_weights())


    env = gym.make('CartPole-v1')
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n

    agent = neuralnet(state_size, action_size)

    scores, episodes = [], []

    for e in range(300):
        done = False
        score = 0
        state = env.reset()
        state = np.reshape(state, [1, state_size])

        while not done:
            if agent.render:
                env.render()
            action = agent.epsilon_greedy(state)
            next_state, reward, done, info = env.step(action)
            next_state = np.reshape(next_state, [1, state_size])
            reward = reward if not done or score == 499 else -100
            agent.sampleadd(state, action, reward, next_state, done)
            agent.learn()
            score += reward
            state = next_state

            if done:
                agent.update_model()
                score = score if score == 500 else score + 100
                scores.append(score)
                episodes.append(e)
                print("episode:", e, "  score:", score, "  epsilon:", agent.epsilon)

                if np.mean(scores[-min(10, len(scores)):]) > 490:
                    sys.exit()
