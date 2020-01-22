import gym
from argparser import Parser
from wrappers import RewardWrapperPlusMinus, ObservationWrapperReshape
from agent import DQNAgent

ENV_NAME = "CartPole-v0"


EXPLORATION_MAX     = 0.01   #set to 1 for new model
EXPLORATION_MIN     = 0.001
EXPLORATION_DECAY   = 0.999

LEARNING_RATE       = 0.001
DISCOUNT_FACTOR     = 0.95

MEMORY_SIZE         = 1000000
BATCH_SIZE          = 32



if __name__== "__main__":
    # create env
    env = gym.make(ENV_NAME).env        #gym.make("Cartpole-v0") sets a limit to 200 turns
                                        #gym.make("Cartpole-v0").env doesn't
    env = RewardWrapperPlusMinus(env)
    env = ObservationWrapperReshape(env)
    # create agent
    agent = DQNAgent(state_space=env.observation_space, action_space=env.action_space,
                     expl_max=EXPLORATION_MAX, expl_min=EXPLORATION_MIN, expl_decay=EXPLORATION_DECAY,
                     learning_rate=LEARNING_RATE, discount_factor=DISCOUNT_FACTOR,
                     memory_size=MEMORY_SIZE)
    # get and parse user args
    args = Parser.parseargs(defaultTrainIterations=10000, defaultEvalIterations=10)
    if args.load:
        agent.load(ENV_NAME, args.loadversion)
    if args.train != 0:
        agent.train(env, args.train, batch_size=32)
    if args.eval != 0:
        agent.evaluate(env, args.eval)
    if args.save:
        agent.save(ENV_NAME, args.saveversion)
    #TODO enable rendering of Cartpole env
    #if args.render:
        #agent.render(env)
    # close env
    env.close()