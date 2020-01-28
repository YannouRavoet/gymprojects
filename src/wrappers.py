import cv2
import gym
from gym import spaces
import numpy as np
from collections import deque


# Maps the reward to +1 for surviving a step and -1 for dying
class RewardNegativeDeath(gym.RewardWrapper):
    def __init__(self, env=None):
        super().__init__(env)

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        return observation, self.reward(reward, done), done, info

    def reward(self, reward, done=False):
        if done:
            return -1
        return 1

# Reshapes the observation to the first box of information
# useful for the cartpole environment
class ObservationReshape(gym.ObservationWrapper):
    def __init__(self, env=None):
        super().__init__(env)

    def observation(self, observation):
        return np.reshape(observation, [1, self.env.observation_space.shape[0]])

# Maps the reward to the sign of the original reward
# useful for the atari games
class RewardSign(gym.RewardWrapper):
    def __init__(self, env=None):
        super().__init__(env)

    def reward(self, reward):
        return np.sign(reward)

# Turns 210x160x3 frames to 1x84x84 frames
# useful for the atari games
class ObservationCHW1x84x84(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        self.observation_space = spaces.Box(low=0, high=255, shape=(1, 84, 84))

    def observation(self, observation):
       return ObservationCHW1x84x84.reshapeframe(observation)

    @staticmethod
    def reshapeframe(frame):
        img = np.reshape(frame, [210, 160, 3]).astype(np.float32)                   #reshape env observation into an 210x160x3 image
        img = cv2.resize(img, (84, 110), interpolation=cv2.INTER_AREA)              #resize to 84x110
        img = img[18:102, :]                                                        #remove borders
        img = img[:, :, 0] * 0.299 + img[:, :, 1] * 0.587 + img[:, :, 2] * 0.114    #grayscale
        img = np.reshape(img, [84, 84, 1])                                          #reshape to 84x84
        img = np.swapaxes(img, 2, 0)                                                #set number of frames as first column
        return img.astype(np.uint8)

class FrameStacker(gym.Wrapper):
    def __init__(self, env, k):
        super().__init__(env)
        self.k = k
        shape = self.observation_space.shape
        self.observation_space = spaces.Box(low=0, high=255, shape=(shape[0]*k, shape[1], shape[2]))
        self.frames = deque([], maxlen=self.k)

    def reset(self):
        obs = self.env.reset()
        for _ in range(self.k):
            self.frames.append(obs)
        return FrameStack(list(self.frames))

    def step(self, action):
        new_state, reward, done, info = self.env.step(action)
        self.frames.append(new_state)
        return FrameStack(list(self.frames)), reward, done, info

class FrameStack(object):
    def __init__(self, frames):
        self.frames = frames

class FireOnDeath(gym.Wrapper):
    def __init__(self, env, fire_index):
        super().__init__(env)
        self.lives  = env.unwrapped.ale.lives()
        self.fire   = fire_index

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        if self.lives > self.env.unwrapped.ale.lives() > 0:
            self.lives = self.env.unwrapped.ale.lives()
            #print(f"death - lives:{self.lives} - firing")
            return self.env.step(self.fire)
        return obs, reward, done, info

    def reset(self):
        self.env.reset()
        self.lives = self.env.unwrapped.ale.lives()
        #print(f"reset - lives:{self.lives} - firing")
        obs, _, _, _ = self.env.step(self.fire)
        return obs


