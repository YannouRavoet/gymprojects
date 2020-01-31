import random
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
from datetime import datetime
class BaseAgent:
    def __init__(self,
                 actions,
                 expl_max, expl_min, expl_decay, model):
        self.e          = expl_max
        self.e_decay    = expl_decay
        self.e_min      = expl_min
        self.actions            = actions      #the number of actions
        self.model              = model

    def act(self, state):
        if np.random.rand() < self.e:
            return random.randrange(self.actions)
        return self.model.predict(state)

    def exploration_decay(self):
            self.e = max(self.e_min, self.e * self.e_decay)

    def save(self, env, version):
        print(f"saving model {version} of {env.unwrapped.spec.id}...")
        self.model.save(env.unwrapped.spec.id, version)

    def load(self,  env, version):
        print(f"loading model {version} of {env.unwrapped.spec.id}...")
        self.model.load(env.unwrapped.spec.id, version)

    def train(self, env, iterations, train_s, save_i):
        print(f"Training for {iterations} iterations with a learning rate of {self.model.lr} and a discount factor of {self.model.gamma}...")
        for i in range(iterations):
            #reset
            state = env.reset()
            done = False
            steps = 0
            score = 0
            #episode
            while not done:
                action = self.act(state)
                steps+=1
                new_state, reward, done, _ = env.step(action)
                score += reward
                if steps%train_s == 0:
                    self.train_update(state, action, new_state, reward, done, i)
                state = new_state
            #save exploration decay and print
            if save_i != 0 and i%save_i==0:
                print(f"saving after iteration {i}")
                self.save(env,"0.0")
            self.exploration_decay()
            print(f"{datetime.now().strftime('%H:%M:%S')} - iteration {i}: score: {score} - steps: {steps} - e: {self.e}")

    #the agent specific training
    def train_update(self, state, action, new_state, reward, done, iteration):
        pass

    def evaluate(self, env, iterations):
        print(f"Evaluating over {iterations} iterations...")
        total_score = 0
        for i in range(iterations):
            state = env.reset()
            done = False
            while not done:
                action = self.model.predict(state)
                new_state, reward, done, _ = env.step(action)
                total_score += reward
                state = new_state
        print(f"Average score over {iterations} iterations: {total_score / iterations}")
        return total_score

    #TODO: add render possibility for text environment Taxi-v3
    #to be used in docker with xvfb-run -s "-screen 0 1400x900x24" python <file.py> -r -rr
    def render_episode(self, env, random_action):
        frames = []
        state = env.reset()
        done = False
        while not done:
            frames.append(env.render(mode='rgb_array'))
            action = env.action_space.sample() if random_action else self.model.predict(state)
            new_state, reward, done, info = env.step(action)
            state = new_state
        frames.append(env.render(mode='rgb_array'))
        plt.figure(figsize=(frames[0].shape[1] / 72.0, frames[0].shape[0] / 72.0), dpi=72)
        patch = plt.imshow(frames[0])
        plt.axis('off')
        def animate(i):
            patch.set_data(frames[i])
        anim = animation.FuncAnimation(plt.gcf(), animate, frames=len(frames), interval=50)
        file_name = env.unwrapped.spec.id + ".gif" if not random_action else env.unwrapped.spec.id + "_rand.gif"
        anim.save(file_name, writer='imagemagick', fps=20)


from models import QTable
class QAgent(BaseAgent):
    def __init__(self,
                 actions,
                 expl_max, expl_min, expl_decay,
                 model):
        #BaseAgent
        super().__init__(actions, expl_max, expl_min, expl_decay, model)

    def train_update(self, state, action, new_state, reward, done,  _):
        self.model.update(state, action, new_state, reward, done)

from collections import deque
class DQNAgent(BaseAgent):
    def __init__(self,
                 actions,
                 expl_max, expl_min, expl_decay,
                 model,
                 memory_size, batch_size):
        #BaseAgent
        super().__init__(actions, expl_max, expl_min, expl_decay, model)
        #DQNAgent
        self.memory     = deque(maxlen=memory_size)
        self.batch_size = batch_size

    def save_experience(self, state, action, new_state, reward, done):
        self.memory.append((state, action, new_state, reward, done))

    #once there are enough experiences in the memory we train a batch after every step
    def train_update(self, state, action, new_state, reward, done, _):
        self.save_experience(state, action, new_state, reward, done)
        if len(self.memory) < self.batch_size:
            return
        batch = random.sample(self.memory, self.batch_size)
        self.model.update_batch(batch)
        self.exploration_decay()

    def init_fill_memory(self, env, experiences):
        print(f"Filling memory with {experiences} random steps of experience...")
        while len(self.memory) < experiences:
            state = env.reset()
            done = False
            while not done:
                action = random.randrange(self.actions)
                new_state, reward, done, _ = env.step(action)
                self.save_experience(state, action, new_state, reward, done)
                state = new_state

TARGET_MODEL_SYNC_ITERATIONS = 50
class DoubleDQN(DQNAgent):
    def __init__(self,
                 actions,
                 expl_max, expl_min, expl_decay,
                 model, target_model,
                 memory_size, batch_size):
        #DQNAgent
        super().__init__(actions, expl_max, expl_min, expl_decay, model, memory_size, batch_size)
        #DQNAgent
        self.target_model = target_model
        self.tau = TARGET_MODEL_SYNC_ITERATIONS
        self.sync_networks()

    def load(self,  env, version):
        super().load(env, version)
        self.sync_networks()

    def sync_networks(self):
        print("Syncing target and training model weights...")
        self.target_model.model.set_weights(self.model.model.get_weights())

    def train(self, env, iterations, train_s, save_i):
        print(f"Training for {iterations} iterations with a learning rate of {self.model.lr} and a discount factor of {self.model.gamma}...")
        for i in range(iterations):
            #reset
            state = env.reset()
            done = False
            steps = 0
            score = 0
            actions=  []
            #episode
            while not done:
                action = self.act(state)
                actions.append(action)
                steps+=1
                new_state, reward, done, _ = env.step(action)
                score += reward
                if steps%train_s == 0:
                    self.train_update(state, action, new_state, reward, done, i)
                self.save_experience(state, action, new_state, reward, done)
                state = new_state
            if i % self.tau == 0:
                self.sync_networks()
            #stats
            unique, counts = np.unique(actions, return_counts=True)
            print(f"{datetime.now().strftime('%H:%M:%S')} - iteration {i}: score: {score} - steps: {steps} - e: {self.e} - actions:{dict(zip(unique, counts))}")
            self.exploration_decay()
            #save if necessary
            if save_i != 0 and i%save_i==0:
                self.save(env,"0.0")
        print("... finished training.")

    def train_update(self, state, action, new_state, reward, done, iteration):
        if len(self.memory) < self.batch_size:
            return
        batch = random.sample(self.memory, self.batch_size)
        self.model.update_batch_with_targetmodel(batch, self.target_model)
        self.exploration_decay()