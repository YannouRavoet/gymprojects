from argparser import Parser
import gym
from wrappers import RewardWrapperSign, ObservationWrapper1x84x84, WrapperFrameStacker
from agent import AtariAgent

ENV_NAME = "Breakout"+"Deterministic-v4"    #Deterministic: skip 4 frames at each step
                                            #v0 25% chance of doing last action instead of new action
                                            #v4 always issue the action that is passed



EXPLORATION_MAX = 1            #set to 1 for new model
EXPLORATION_MIN = 0.001
EXPLORATION_DECAY = 0.99       #higher is slower decay

LEARNING_RATE   = 0.1
DISCOUNT_FACTOR = 0.99


MEMORY_SIZE = 1000000
BATCH_SIZE  = 32

MODEL_SAVE_EVERY = 10

if __name__== "__main__":
    # create env
    env = gym.make(ENV_NAME).env
    env = RewardWrapperSign(env)
    env = ObservationWrapper1x84x84(env)
    env = WrapperFrameStacker(env, 4)
    # create agent
    agent = AtariAgent(obs_space_shp=env.observation_space, actions=env.action_space,
                       expl_max=EXPLORATION_MAX, expl_min=EXPLORATION_MIN, expl_decay=EXPLORATION_DECAY,
                       learning_rate=LEARNING_RATE, discount_factor=DISCOUNT_FACTOR,
                       memory_size=MEMORY_SIZE, batch_size=32)
    # get and parse user args
    args = Parser.parseargs(defaultTrainIterations=10000, defaultEvalIterations=10)
    if args.load:
        agent.load(env, args.loadversion)
    if args.train != 0:
        agent.train(env, args.train, save_i=MODEL_SAVE_EVERY)
    if args.eval != 0:
        print("Evaluation results (higher scores are better):")
        agent.evaluate(env, args.eval)
    if args.save:
        agent.save(env, args.saveversion)
    #TODO enable rendering of Breakout env
    #if args.render:
        #agent.render(env)
    # close env
    env.close()