import random
import numpy as np


class BaseAgent:
    def __init__(self,
                 obs_space_shp, actions,
                 expl_max, expl_min, expl_decay):
        self.e          = expl_max
        self.e_decay    = expl_decay
        self.e_min      = expl_min
        self.obs_space_shape    = obs_space_shp         #the number of states
        self.actions            = actions      #the number of actions
        self.model              = None

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

    def train(self, env, iterations, save_i):
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
                if i%4 == 0: #keep enough variation over the
                    self.train_update(state, action, new_state, reward, done, i)
                state = new_state
            print(f"iteration {i}: score: {score} - steps: {steps} - e: {self.e}")
            self.exploration_decay()
            #save if necessary
            if save_i != 0 and i%save_i==0:
                print(f"saving after iteration {i}")
                self.save(env,"0.0")
        print("... finished training.")

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

    def random_episodes(self, env, iterations):
        steps = 0
        for _ in range(iterations):
            env.reset()
            done = False
            #episode
            while not done:
                _, _, done, _ = env.step(random.randrange(self.actions))
                steps+=1
        print(f"Averaged {steps/iterations} steps per iteration over {iterations} iterations")

    def render(self,env):
        state = env.reset()
        done = False
        while not done:
            env.render()
            action = self.model.predict(state)
            new_state, reward, done, info = env.step(action)
            state = new_state
        env.render()

from models import QTable
class QAgent(BaseAgent):
    def __init__(self,
                 obs_space_shp, actions,
                 expl_max, expl_min, expl_decay,
                 learning_rate, discount_factor):
        #BaseAgent
        super().__init__(obs_space_shp, actions, expl_max, expl_min, expl_decay)
        #QAgent
        self.model = QTable(nostates=obs_space_shp, noactions=actions,
                            learning_rate=learning_rate, discount_factor=discount_factor)

    def train_update(self, state, action, new_state, reward, done,  _):
        self.model.update(state, action, new_state, reward, done)

from collections import deque
class DQNBaseAgent(BaseAgent):
    def __init__(self,
                 obs_space_shp, actions,
                 expl_max, expl_min, expl_decay,
                 memory_size, batch_size):
        #BaseAgent
        super().__init__(obs_space_shp, actions, expl_max, expl_min, expl_decay)
        #DQNAgent
        self.memory     = deque(maxlen=memory_size)
        self.model      = None
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

from models import CartpoleNetwork
class CartPoleAgent(DQNBaseAgent):
    def __init__(self,
                 obs_space_shp, actions,
                 expl_max, expl_min, expl_decay,
                 learning_rate, discount_factor,
                 memory_size, batch_size):
        #DQNBaseAgent
        super().__init__(obs_space_shp, actions, expl_max, expl_min, expl_decay, memory_size, batch_size)
        #CartPole Agent
        self.model = CartpoleNetwork(learning_rate, discount_factor,
                                     input_shape=(self.obs_space_shape,), output_shape=self.actions)


from models import AtariNetwork
TARGET_MODEL_SYNC_ITERATIONS = 4000
class AtariAgent(DQNBaseAgent):
    def __init__(self,
                 obs_space_shp, actions,
                 expl_max, expl_min, expl_decay,
                 learning_rate, discount_factor,
                 memory_size, batch_size):
        #DQNBaseAgent
        super().__init__(obs_space_shp.shape, actions.n, expl_max, expl_min, expl_decay,memory_size, batch_size)
        #DQNAgent
        self.model = AtariNetwork(learning_rate, discount_factor,
                                  input_shape=self.obs_space_shape, output_shape=self.actions)
        self.target_model = AtariNetwork(learning_rate, discount_factor, input_shape=self.obs_space_shape, output_shape=self.actions)
        self.tau = TARGET_MODEL_SYNC_ITERATIONS
        self.sync_networks()

    def load(self,  env, version):
        super().load(env, version)
        self.sync_networks()

    def sync_networks(self):
        self.target_model.model.set_weights(self.model.model.get_weights())


    #once there are enough experiences in the memory we train a batch after every step
    def train_update(self, state, action, new_state, reward, done, iteration):
        self.save_experience(state, action, new_state, reward, done)
        if len(self.memory) < self.batch_size:
            return
        batch = random.sample(self.memory, self.batch_size)
        self.model.update_batch_with_targetmodel(batch, self.target_model)
        if iteration % self.tau == 0:
            self.sync_networks()
        self.exploration_decay()