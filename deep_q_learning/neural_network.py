"""Document defining the Deep Q network
"""
from collections import deque
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
import numpy as np
import os
import random

# Seeding Random for replicable result
np.random.seed(2017)
# random.seed(2017)

class neuralNetwork:
    """This class contains the following:
    """
    # Verbosity toggle
    verbose = False
    # Neural Network Hyper Parameters
    learning_rate = 0.001
    BATCH_SIZE = 20
    # Q-Learning Parameters
    EPSILON_MAX = 1.0
    EPSILON_MIN = 0.01
    EPSILON_DECAY = 0.995
    GAMMA = 0.95
    # Defining Training Database
    MEMORY_SIZE = 1000000

    def __init__(self, action_space, observation_space):
        """Initialization of neural network.
        """ 
        self.action_space = action_space
        self.observation_space = observation_space
        self.epsilon = self.EPSILON_MAX
        # Defining the Neural Network
        self.model = Sequential()
        self.model.add(Dense(24, 
                        input_shape=observation_space,
                        activation='relu',
                        kernel_initializer='random_uniform',
                        bias_initializer='zeros'))
        self.model.add(Dense(24,
                        activation='relu',
                        kernel_initializer='random_uniform',
                        bias_initializer='zeros'))
        self.model.add(Dense(self.action_space,
                        activation='linear',
                        kernel_initializer='random_uniform',
                        bias_initializer='zeros'))
        self.model.compile(loss='mse',
                            optimizer=Adam(lr=self.learning_rate))
        if (self.verbose):
            msg = "The Neural Network has been initialized." + \
                " The Network definition is:\n"
            print(msg)
            print(self.model.summary())
        # Initializing Memory to store training data
        self.memory = deque(maxlen=self.MEMORY_SIZE)
        if (self.verbose):
            msg = "\nMemory to store training data is initialized." + \
                " Memory can store {} instances. \n".format(self.MEMORY_SIZE)
            print(msg)
    
    def act(self, state, env):
        """Function to take actions.
        """
        if np.random.uniform(low=0.0, high=1.0) < self.epsilon:
            # Exploration
            if (self.verbose):
                msg = "Exploration Step, random action taken.\n"
                print(msg)
            return env.action_space.sample()
        else:
            # Exploitation
            Q_values = self.model.predict(state)
            if (self.verbose):
                msg = "Exploitation Step. Argmax of NN output chosen.\n"
                print(msg)
            return np.argmax(Q_values[0])
    
    def remember(self, state, action, reward, next_state, done):
        """Record the data into memory
        """
        self.memory.append((state, action, reward, next_state, done))
        if (self.verbose):
            msg = "Appended Data to memory.\n"
            print(msg)
    
    def learn(self):
        """Function to learn from memory"""
        # Collecting data to train
        if (len(self.memory) < self.BATCH_SIZE):
            if (self.verbose):
                msg = "Memory < batch size! Training Terminated.\n"
                print(msg)
            return
        batch = random.sample(self.memory, self.BATCH_SIZE)
        # Training
        for state, action, reward, next_state, terminate in batch:
            Q_update = reward
            if not terminate:
                Q_update = (reward 
                            + self.GAMMA 
                            * np.amax(self.model.predict(next_state)[0]))
            Q_values = self.model.predict(state)
            Q_values[0][action] = Q_update
            self.model.fit(state, Q_values, verbose=0)
        # Updating the exploration rate
        self.epsilon = self.epsilon * self.EPSILON_DECAY
        self.epsilon = max(self.EPSILON_MIN, self.epsilon)
        if (self.verbose):
            msg = "Training based on experience Done.\n"
            print(msg)

    def save(self, path):
        """Function to save and delete the model"""
        path = os.path.join(path, "networkModel.h5")
        self.model.save(path)
        del self.model
        if (self.verbose):
            msg = "Model saved as: {}\n".format(path)
            print(msg)