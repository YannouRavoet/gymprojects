#RMSPROP
LEARNING_RATE = 0.00025
RHO=0.95
EPSILON=0.01

class BaseModel:
    def __init__(self,
                 learning_rate, discount_factor):
        self.lr     = learning_rate
        self.gamma  = discount_factor

    def predict(self, state):
        pass

    def save(self, env_name, version):
        pass

    def load(self, env_name, version):
        pass

import numpy as np
class QTable(BaseModel):
    def __init__(self,
                 nostates, noactions,
                 learning_rate, discount_factor):
        #BaseModel
        super().__init__(learning_rate, discount_factor)
        #QTable
        self.model = np.zeros([nostates, noactions])

    def predict(self, state):
        return np.argmax(self.model[state])

    def update(self, state, action, new_state, reward, _):
        q_state_action = self.model[state, action]      #Current utility of this state
        qmax_new_state = np.max(self.model[new_state])  #Expected utility of next state
        new_value = (1 - self.lr) * q_state_action + self.lr * (reward + self.gamma * qmax_new_state)
        self.model[state, action] = new_value

    def save(self,env_name, version):
        np.savetxt('./models/' + env_name + '/' + 'taxi_qtable_'+version+'.csv', self.model, delimiter=',')

    def load(self, env_name, version):
        self.model = np.loadtxt('./models/' + env_name + '/' + 'taxi_qtable_'+version+'.csv', delimiter=',')

class NeuralNetwork(BaseModel):
    def __init__(self,
                 learning_rate, discount_factor,
                 input_shape, output_shape):
        #BaseModel
        super().__init__(learning_rate, discount_factor)
        #NeuralNetwork
        self.model = None #implementation in specific network class

    def predict(self, state):
        return np.argmax(self.model.predict(state)[0])

    def update(self, batch):
        states = []
        q_values = []
        for state, action, new_state, reward, done in batch:
            states.append(state)
            #calculate the update
            q_update = reward
            if not done:
                q_update = (reward + self.gamma * np.max(self.predict(new_state)))
            #update the q_value of the current state action pair with prediction
            q_value_state = self.model.predict(state)
            q_value_state[0][action] = q_update
            q_values.append(q_value_state)
        self.model.fit(np.asarray(states).squeeze(), np.asarray(q_values).squeeze(), batch_size=len(batch),verbose=0)

    def save(self, env_name, version):
        self.model.save_weights('./models/' + env_name + '/' + version)

    def load(self, env_name, version):
        self.model.load_weights('./models/'+ env_name + '/' + version)

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
class CartpoleNetwork(NeuralNetwork):
    def __init__(self,
                 learning_rate, discount_factor,
                 input_shape, output_shape):
        #NeuralNetwork
        super().__init__(learning_rate, discount_factor, input_shape, output_shape)
        #CartpoleNetwork
        self.model = Sequential()
        self.model.add(Dense(24, input_shape=input_shape, activation='relu'))
        self.model.add(Dense(24, activation='relu'))
        self.model.add(Dense(output_shape, activation='linear'))
        self.model.compile(loss='mse', optimizer=Adam(lr=learning_rate))
        self.model.summary()

import tensorflow as tf
from tensorflow.keras.layers import Conv2D, Flatten
from tensorflow.keras.optimizers import RMSprop
class AtariNetwork(NeuralNetwork):
    # Solution to 'Failed to get convolution algorithm' 'Could not create cudnn handle: CUDNN_STATUS_INTERNAL_ERROR'
    # Might be due to local setup: RTX 2070
    def _quickfixError(self):
        gpus = tf.config.experimental.list_physical_devices('GPU')
        if gpus:
            try:
                # Currently, memory growth needs to be the same across GPUs
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
                logical_gpus = tf.config.experimental.list_logical_devices('GPU')
                print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
            except RuntimeError as e:
                print(e)

    def __init__(self, learning_rate, discount_factor, input_shape, output_shape):
        self._quickfixError()
        #NeuralNetwork
        super().__init__(learning_rate, discount_factor, input_shape, output_shape)
        #AtariNetwork
        self.model = Sequential()
        self.model.add(Conv2D(32,
                         8,
                         strides=(4, 4),
                         padding="valid",
                         activation="relu",
                         input_shape=input_shape,
                         data_format="channels_first"))
        self.model.add(Conv2D(64,
                         4,
                         strides=(2, 2),
                         padding="valid",
                         activation="relu",
                         input_shape=input_shape,
                         data_format="channels_first"))
        self.model.add(Conv2D(64,
                         3,
                         strides=(1, 1),
                         padding="valid",
                         activation="relu",
                         input_shape=input_shape,
                         data_format="channels_first"))
        self.model.add(Flatten())
        self.model.add(Dense(512, activation="relu"))
        self.model.add(Dense(output_shape))
        self.model.compile(loss="mean_squared_error",
                      optimizer=RMSprop(lr=LEARNING_RATE,
                                        rho=RHO,
                                        epsilon=EPSILON),
                      metrics=["accuracy"])
        self.model.summary()

    def update_with_targetmodel(self, batch, targetmodel):
        states = []
        q_values = []
        for state, action, new_state, reward, done in batch:
            states.append(state)
            # calculate the update
            q_update = reward
            if not done:
                q_update = (reward + self.gamma * np.max(targetmodel.predict(new_state)))
            # update the q_value of the current state action pair with prediction
            q_value_state = self.model.predict(state)
            q_value_state[0][action] = q_update
            q_values.append(q_value_state)
        self.model.fit(np.asarray(states).squeeze(), np.asarray(q_values).squeeze(), batch_size=len(batch), verbose=0)