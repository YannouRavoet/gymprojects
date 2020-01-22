import cv2
import gym
from gym import spaces
import numpy as np
from collections import deque


# Maps the reward to +1 for surviving a step and -1 for dying
class RewardWrapperPlusMinus(gym.RewardWrapper):
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
class ObservationWrapperReshape(gym.ObservationWrapper):
    def __init__(self, env=None):
        super().__init__(env)

    def observation(self, observation):
        return np.reshape(observation, [1, self.env.observation_space.shape[0]])

# Maps the reward to the sign of the original reward
# useful for the atari games
class RewardWrapperSign(gym.RewardWrapper):
    def __init__(self, env=None):
        super().__init__(env)

    def reward(self, reward):
        return np.sign(reward)

# Turns 210x160x3 frames to 1x84x84 frames
# useful for the atari games
class ObservationWrapper1x84x84(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        self.observation_space = spaces.Box(low=0, high=255, shape=(1, 84, 84))

    def observation(self, observation):
       return ObservationWrapper1x84x84.reshapeframe(observation)

    @staticmethod
    def reshapeframe(frame):
        img = np.reshape(frame, [210, 160, 3]).astype(np.float32)                   #reshape env observation into an 210x160x3 image
        img = cv2.resize(img, (84, 110), interpolation=cv2.INTER_AREA)              #resize to 84x110
        img = img[18:102, :]                                                        #remove borders
        img = img[:, :, 0] * 0.299 + img[:, :, 1] * 0.587 + img[:, :, 2] * 0.114    #grayscale
        img = np.reshape(img, [84, 84, 1])                                          #reshape to 84x84
        img = np.swapaxes(img, 2, 0)                                                #set number of frames as first column
        return img.astype(np.uint8)

class WrapperFrameStacker(gym.Wrapper):
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

    def __array__(self, dtype=None):
        out = np.concatenate(self.frames, axis=0)
        if dtype is not None:
            out = out.astype(dtype)
        return out