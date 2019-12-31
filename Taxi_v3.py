import gym
import random
import numpy as np

ENV_NAME = "Taxi-v3"


GAMMA = 0.6
LEARNING_RATE = 0.1
EXPLORATION_MAX = 0.1
EXPLORATION_DECAY = 0.999
EXPLORATION_MIN = 0.01

TRAINING_ITERATIONS = 100000

class Q_agent:
    def __init__(self, observation_space, action_space):
        self.exploration_rate = EXPLORATION_MAX              #epsilon goes down as we train
        self.observation_space = observation_space
        self.action_space = action_space
        self.q_table = np.zeros([self.observation_space, self.action_space])

    def train(self, env):
        for i in range(TRAINING_ITERATIONS):
            state = env.reset()
            done = False
            while not done:
                action = self.act(state)
                next_state, reward, done, info = env.step(action)
                #update
                old_value = self.q_table[state, action]
                next_max = np.max(self.q_table[next_state])
                new_value = (1 - LEARNING_RATE) * old_value + LEARNING_RATE * (reward + GAMMA * next_max)
                self.q_table[state, action] = new_value

                state = next_state
            self.exploration_rate = max(EXPLORATION_MIN, self.exploration_rate * EXPLORATION_DECAY)
        print("Training finished...")
        self.save_qtable()
        print("Q-Table saved...")

    def act(self, state):
        if np.random.rand() < self.exploration_rate:
            return random.randrange(self.action_space)
        return np.argmax(self.q_table[state])

    def save_qtable(self):
        np.savetxt('./models/'+ ENV_NAME + '/taxi_qtable.csv', self.q_table, delimiter=',')
    def load_qtabel(self):
        self.q_table = np.loadtxt('./models/'+ ENV_NAME + '/taxi_qtable.csv', delimiter=',')

    def evaluate(self, env, times=10):
        total_score = 0
        for i in range(times):
            state = env.reset()
            done = False
            while not done:
                if i == 0:
                    env.render()
                action = np.argmax(self.q_table[state])
                next_state, reward, done, info = env.step(action)
                total_score += reward
                state = next_state
            if i==0:
                env.render()
        print(f"Average score over {times} iterations: {total_score/times}")

def taxi_v3():
    env = gym.make(ENV_NAME)
    env.reset()
    agent = Q_agent(env.observation_space.n, env.action_space.n)
    agent.load_qtabel()
    agent.train(env)
    agent.evaluate(env, times = 10)
    env.close()


if __name__=='__main__':
    taxi_v3()