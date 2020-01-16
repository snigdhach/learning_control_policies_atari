import gym
import math
from copy import deepcopy
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import random

random.seed(2017)
env = gym.make('CartPole-v0')
# Q_table = np.zeros((2,2,8,4,2))
Q_table= np.zeros((2,2,8,4,2))
alpha=0.1
# buckets=[2, 2, 8, 4] #Bucket values 
buckets = [2,2,8,4]
gamma=0.5
rewards=[] #TO get Reward values
reward_check = []
no_episodes = 2000 #number of episodes training
flag = 0 #for checking first episode
count = 0 # to count number of wins in number # of episodes
max_range = []

for i in range(4):
    # max_range.append(float((random.randint(1,5))/10))
    max_range.append(random.randint(1,5))
print(max_range)    
#Discretizing the State
def Discretize(observation):
    # print("I am in descritize)")
    # print(len(observation))
    interval=[0 for i in range(len(observation))]
    # max_range=[2,3,0.42,3]
    # max_range=[3,3,3,3]


    for i in range(len(observation)):
        data = observation[i]
        inter = int(math.floor((data + max_range[i])/(2*max_range[i]/buckets[i])))
        if inter>=buckets[i]:
            interval[i]=buckets[i]-1
        elif inter<0:
            interval[i]=0
        else:
            interval[i]=inter
    return interval
	


def get_action(observation,t):
#     print get_explore_rate(t)
    if np.random.random()<max(0.05, min(0.5, 1.0 - math.log10((t+1)/150.))):
       return env.action_space.sample()
    interval = Discretize(observation)

    # if Q_table[tuple(interval)][0] >=Q_table[tuple(interval)][1]:
    #     return 0
    # else:
    #     return 1
    return np.argmax(np.array(Q_table[tuple(interval)]))

def Sarsa_Update(observation,reward,action,ini_obs,next_action,t):
    interval = Discretize(observation)
    Q_next = Q_table[tuple(interval)][next_action]
    Discretized_value = Discretize(ini_obs)
    # print("I am here")
    Q_table[tuple(Discretized_value)][action]+=alpha*(reward + gamma*(Q_next) - Q_table[tuple(Discretized_value)][action])
    # Sarsa Formula  Q(st,at) = Q(st,at) + alpha [ reward + gamma*(Q(s(t+1),a(t+1))) - Q(st,at)]
		
for i_episode in range(no_episodes+1):
    observation = env.reset()
	#done = False;
    t=0
    # print(i_episode)
    while (True):
        env.render()
        action = get_action(observation,i_episode)
        observation_new, reward, done, info = env.step(action)
        # print("observation_new, reward, done, info ", observation_new, reward, done, info)
        next_action = get_action(observation_new,i_episode)
        Sarsa_Update(observation_new,reward,action,observation,next_action,t)
        observation=observation_new
        action = next_action
        t+=1
        if t+1 >=195 and done:
            count = count +1 
        if t+1 >=195 and flag ==0:
            print("First win at Episode ", i_episode, " with Reward Score - ", t+1)
            flag=1
        if done: 
            #t+1 being used as episode starts from 0
            print("Episode ", i_episode, " Reward Score - ", t+1)
        if done:
#             print("Episode finished after {} timesteps".format(t+1))
            rewards.append(t+1)
            break
# print(rewards)
print("Total number of wins " , count ," in ", no_episodes, " Episodes ")
plt.plot(rewards)
plt.show()