import sys
import gym
import pylab
import numpy as np
import matplotlib.pyplot as plt
from keras.layers import Dense
from keras.models import Sequential
from keras.optimizers import Adam

EPISODES = 2000


#Actor Critic model for Reinforcement Learning
class ActorCritic:
    def __init__(self, state_size, action_size):
        # True on False if you we want to see or not
        self.render = True
        #Since Deep learning takes time, saving model
        self.load_model = False
        # Getting the Action and state
        self.state_size = state_size
        self.action_size = action_size

        # These are hyper parameters for the Policy Gradient
        self.discount_factor = 0.99
        self.LearningRate_Actor = 0.001
        self.LearningRate_Critic = 0.005

        # create model for policy network
        self.actor = self.Actor()
        self.critic = self.Critic()

        if self.load_model:
            self.actor.load_weights("./cartpole_actor.h5")
            self.critic.load_weights("./cartpole_critic.h5")

    # actor: state is input and probability of each action is output of model
    def Actor(self):
        actor = Sequential()
        actor.add(Dense(24, input_dim=self.state_size, activation='relu',
                        kernel_initializer='he_uniform'))
        actor.add(Dense(self.action_size, activation='softmax',
                        kernel_initializer='he_uniform'))
        actor.summary() 
        #Categorical cross entropy is defined H(p, q) = sum(p_i * log(q_i)). 
        #H(p, q) = A * log(policy(s, a)) in our case as all other probs are 0
        actor.compile(loss='categorical_crossentropy',
                      optimizer=Adam(lr=self.LearningRate_Actor))
        return actor

    # critic: state is input and value of state is reward that we expect.
    def Critic(self):
        critic = Sequential()
        critic.add(Dense(24, input_dim=self.state_size, activation='relu',
                         kernel_initializer='he_uniform'))
        critic.add(Dense(1, activation='linear',
                         kernel_initializer='he_uniform'))
        critic.summary()
        critic.compile(loss="mse", optimizer=Adam(lr=self.LearningRate_Critic))
        return critic

    # using the output of policy network, pick action stochastically
    def get_action(self, state):
        action = self.actor.predict(state, batch_size=1).flatten()
        # print(policy)
        return np.random.choice(self.action_size, 1, p=action)[0]

    # update policy network every episode
    def train_model(self, state, action, reward, next_state, done):
        # expected_reward = np.zeros((1, self.value_size))
        expected_reward = np.zeros((1, 1))
        expected_action = np.zeros((1, self.action_size))

        #Predict Reward for current states and next_state
        value = self.critic.predict(state)[0]
        next_value = self.critic.predict(next_state)[0]

        if done:
            expected_action[0][action] = reward - value
            expected_reward[0][0] = reward
        else:
            expected_action[0][action] = reward + self.discount_factor * (next_value) - value
            expected_reward[0][0] = reward + self.discount_factor * next_value

        self.actor.fit(state, expected_action, epochs=1, verbose=0)
        self.critic.fit(state, expected_reward, epochs=1, verbose=0)


if __name__ == "__main__":
    env = gym.make('CartPole-v0')
    # get size of state and action from environment
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n

   	#ActorCritic Agent Definition
    agent = ActorCritic(state_size, action_size)

    scores, episodes = [], []
    flag = 0;
    count = 0;
    for e in range(EPISODES):
        done = False
        score = 0
        state = env.reset()
        state = np.reshape(state, [1, state_size])
        while not done:
            if agent.render:
                env.render()

            action = agent.get_action(state)
            next_state, reward, done, info = env.step(action)
            next_state = np.reshape(next_state, [1, state_size])
            if not done or score ==195:
            	reward = reward 
            else:
            	reward = -100
            agent.train_model(state, action, reward, next_state, done)

            score += reward
            state = next_state

            if done:
                score = score if score == 195.0 else score + 100
                if score >=195:
                	count = count + 1
                if flag ==0 and score >= 195 : 
                	print("First win at ", e+1 , " with score ", score)
                	flag = 1;

                scores.append(score)
                episodes.append(e)
                pylab.plot(episodes, scores, 'b')
                pylab.savefig("./cartpole_a2c.png")
                print("episode:", e, "  score:", score)
    print(count , " number of wins in ", EPISODES , " Episodes. ")
    plt.plot(scores)
    plt.show()      
agent.actor.save_weights("./cartpole_actor.h5")
agent.critic.save_weights("./cartpole_critic.h5")
