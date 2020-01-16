import gym
import numpy as np
from image_preproces import transformImage

env = gym.make('CarRacing-v0')
obs = env.reset()
print (np.shape(obs))
IMG = transformImage()
# IMG.display_image(obs)
# i = IMG.grayscale(obs)
# pca = IMG.pca_compression(obs)
# print (pca.shape)