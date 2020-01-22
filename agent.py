import random
import numpy as np


class BaseAgent:
    def __init__(self,
                 state_space, action_space,
                 expl_max, expl_min, expl_decay):
        self.e          = expl_max
        self.e_decay    = expl_decay
        self.e_min      = expl_min
        self.state_space    = state_space       #the number of states
        self.action_space   = action_space      #the number of actions
        self.model          = None

    def act(self, state):
        if np.random.rand() < self.e:
            return random.randrange(self.action_space)
        return self.predict(state)
    def predict(self, state):
        return self.model.predict(state)
    def exploration_decay(self):
            self.e = max(self.e_min, self.e * self.e_decay)
    def save(self, env_name, version):
        print(f"saving model {version}...")
        self.model.save(env_name, version)
    def load(self, env_name,  version):
        print(f"loading model {version}...")
        self.model.load(env_name, version)

    def train(self, env, iterations):
        pass
    def evaluate(self, env, iterations):
        pass

    def render(self,env):
        state = env.reset()
        done = False
        while not done:
            env.render()
            action = self.predict(state)
            new_state, reward, done, info = env.step(action)
            state = new_state
        env.render()



from models import QTable
class QAgent(BaseAgent):
    def __init__(self,
                 state_space, action_space,
                 expl_max, expl_min, expl_decay,
                 learning_rate, discount_factor):
        #BaseAgent
        super().__init__(state_space.n, action_space.n, expl_max, expl_min, expl_decay)
        #QAgent
        self.model = QTable(nostates=state_space.n, noactions=action_space.n,
                            learning_rate=learning_rate,discount_factor=discount_factor)

    def train(self, env, iterations):
        print(f"Training for {iterations} iterations...")
        for i in range(iterations):
            if i%1000==0:
                print(f"iterations {i}")
            state = env.reset()
            done = False
            while not done:
                action = self.act(state)
                new_state, reward, done, info = env.step(action)
                self.model.update(state, action, new_state, reward, done, info)
                state = new_state
            self.exploration_decay()
        print("... finished training.")

    def evaluate(self, env, iterations):
        total_score = 0
        for i in range(iterations):
            state = env.reset()
            done = False
            while not done:
                action = self.predict(state)
                new_state, reward, done, info = env.step(action)
                total_score += reward
                state = new_state

        print("Evalutation results (lower scores are better):")
        print(f"Average score over {iterations} iterations: {total_score / iterations}")


from collections import deque
from models import CartpoleNetwork
class DQNAgent(BaseAgent):
    def __init__(self,
                 state_space, action_space,
                 expl_max, expl_min, expl_decay,
                 learning_rate, discount_factor,
                 memory_size):
        #BaseAgent
        super().__init__(state_space.shape[0], action_space.n, expl_max, expl_min, expl_decay)
        #DQNAgent
        self.memory = deque(maxlen=memory_size)
        self.model = CartpoleNetwork(learning_rate, discount_factor, input_shape=(self.state_space,), output_shape=self.action_space)

    def save_experience(self, state, action,new_state, reward, done, info):
        self.memory.append((state, action, new_state, reward, done, info))

    def train(self, env, iterations, batch_size=32):
        print(f"Training for {iterations} iterations...")
        for i in range(iterations):
            state = env.reset()
            steps=0
            done = False
            while not done:
                action = self.act(state)
                new_state, reward, done, info = env.step(action)
                steps +=1
                self.save_experience(state, action, new_state, reward, done, info)
                state = new_state
                self.batch_train(batch_size)
            if i%1==0:
                print(f"iteration {i}: steps={steps}")
        print("... training finished.")

    def batch_train(self, batch_size):
        if len(self.memory) < batch_size:
            return
        batch = random.sample(self.memory, batch_size)
        for state, action, new_state, reward, done, info in batch:
            self.model.update(state, action, new_state, reward, done, info)
        self.exploration_decay()

    def evaluate(self, env, iterations):
        total_score = 0
        for i in range(iterations):
            state = env.reset()
            done = False
            while not done:
                env.render()
                action = self.predict(state)
                new_state, reward, done, info = env.step(action)
                total_score += reward
                state = new_state
        print("Evalutation results (higher scores are better):")
        print(f"Average score over {iterations} iterations: {total_score / iterations}")

from models import AtariNetwork
class DDQNAgent(BaseAgent):
    def __init__(self,
                 state_space, action_space,
                 expl_max, expl_min, expl_decay,
                 learning_rate, discount_factor,
                 memory_size):
        #BaseAgent
        super().__init__(state_space.shape, action_space.n, expl_max, expl_min, expl_decay)
        #DQNAgent
        self.memory = deque(maxlen=memory_size)
        self.model = AtariNetwork(learning_rate, discount_factor, input_shape=self.state_space, output_shape=self.action_space)

    def predict(self, state):
        reshapedstate = np.expand_dims(np.asarray(state).astype(np.float64), axis=0)
        return self.model.predict(reshapedstate)

    def save_experience(self, state, action,new_state, reward, done, info):
        self.memory.append((state, action, new_state, reward, done, info))

    def train(self, env, iterations, batch_size=32):
        print(f"Training for {iterations} iterations...")
        for i in range(iterations):
            state = env.reset()
            steps=0
            done = False
            while not done:
                action = self.act(state)
                new_state, reward, done, info = env.step(action)
                steps +=1
                self.save_experience(state, action, new_state, reward, done, info)
                state = new_state
                self.batch_train(batch_size)
            if i%1==0:
                print(f"iteration {i}: steps={steps}")
        print("... training finished.")

    def batch_train(self, batch_size):
        if len(self.memory) < batch_size:
            return
        batch = random.sample(self.memory, batch_size)
        for state, action, new_state, reward, done, info in batch:
            self.model.update(state, action, new_state, reward, done, info)
        self.exploration_decay()

    def evaluate(self, env, iterations):
        total_score = 0
        for i in range(iterations):
            state = env.reset()
            done = False
            while not done:
                env.render()
                action = self.predict(state)
                new_state, reward, done, info = env.step(action)
                total_score += reward
                state = new_state
        print("Evalutation results (higher scores are better):")
        print(f"Average score over {iterations} iterations: {total_score / iterations}")
