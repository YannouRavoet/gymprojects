import gym
import random
import numpy as np
from collections import deque
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
print(f"tensorflow version: {tf.__version__}")
#TODO conda install tensorflow-gpu==2.1.0rc2 once out in order to fix memory leak with tf.predict() and tf.fit()
#Current fix = running on CPU with pip install tensorflow-gpu==2.1.0rc2
print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))

ENV_NAME = "CartPole-v0"

GAMMA = 0.95
LEARNING_RATE = 0.001

MEMORY_SIZE = 1000000
BATCH_SIZE = 32

EXPLORATION_MAX = 0.01   #set to 1 for new model
EXPLORATION_MIN = 0.001
EXPLORATION_DECAY = 0.999

TRAIN_ITERATIONS = 200000
EVALUATION_ITERATIONS = 10

class DQNSolver:
    def __init__(self, observation_space, action_space):
        self.exploration_rate = EXPLORATION_MAX
        self.observation_space = observation_space
        self.action_space = action_space
        self.memory = deque(maxlen=MEMORY_SIZE) #drops the earliest experiences when it reaches maxlen

        self.model = Sequential()
        self.model.add(Dense(24, input_shape=(observation_space,), activation='relu'))
        self.model.add(Dense(24, activation='relu'))
        self.model.add(Dense(self.action_space, activation='linear'))

        self.model.compile(loss='mse', optimizer=Adam(lr=LEARNING_RATE))

    def save_model(self, model_version = 'default'):
        self.model.save_weights('./models/'+ ENV_NAME + '/' + model_version)
        print("model saved...")
    def load_model(self, model_version = 'default'):
        self.model.load_weights('./models/'+ ENV_NAME + '/' + model_version)
        print("model loaded...")

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() <self.exploration_rate:
            return random.randrange(self.action_space)
        q_values = self.model.predict(state)
        return np.argmax(q_values[0])

    def experience_replay(self):
        if len(self.memory) < BATCH_SIZE:
            return
        batch = random.sample(self.memory, BATCH_SIZE)
        for state, action, reward, next_state, done in batch:
            q_update = reward
            if not done:
                q_update = (reward + GAMMA * np.max(self.model.predict(next_state) [0]))
            q_values = self.model.predict(state)
            q_values[0][action] = q_update
            self.model.fit(state, q_values, verbose=0)
        self.exploration_rate = max(EXPLORATION_MIN,  self. exploration_rate * EXPLORATION_DECAY)

    def evaluate(self, env, render = False):
        scores = []
        for iteration in range(EVALUATION_ITERATIONS):
            done = False
            total_reward = 0
            state = env.reset()
            state = np.reshape(state, [1,self.observation_space])
            while not done:
                if render:
                    env.render()
                action = np.argmax(self.model.predict(state)[0])
                next_state, reward, done, info = env.step(action)
                next_state = np.reshape(next_state, [1,self.observation_space])
                reward = reward if not done else -reward
                total_reward += reward
                state = next_state
            print(f"iteration {iteration}: {total_reward}")
            scores.append(total_reward)
        print(f"Average score over {EVALUATION_ITERATIONS} iterations = {np.average(scores)}")

def cartpole():
    env = gym.make(ENV_NAME).env        #gym.make("Cartpole-v0") sets a limit to 200 turns
    observation_space = env.observation_space.shape[0]
    action_space = env.action_space.n
    agent = DQNSolver(observation_space, action_space)
    agent.load_model()
    if True:
        scores = []
        for i in range(TRAIN_ITERATIONS):
            state = env.reset()
            state = np.reshape(state, [1,observation_space])
            done = False
            total_reward = 0
            while not done:
                action = agent.act(state)
                next_state, reward, done, info = env.step(action)
                reward = reward if not done else -reward            #reward adaptation: death = penalty
                total_reward += reward
                next_state = np.reshape(next_state, [1, observation_space])
                agent.remember(state, action, reward, next_state, done)
                state = next_state
                agent.experience_replay()
            #END WHILE NOT DONE
            scores.append(total_reward)
            print(f"iteration {i}: e={agent.exploration_rate} - iteration_score={total_reward} - average={np.average(scores)}")
            if (i+1)%5==0:
                agent.save_model()
        #END FOR ITERATIONS
        agent.save_model()
    agent.evaluate(env, render = True)
    env.close()


if __name__== "__main__":
    cartpole()