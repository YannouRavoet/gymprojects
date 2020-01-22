import gym
from argparser import Parser
from agent import QAgent

ENV_NAME = "Taxi-v3"

EXPLORATION_MAX         = 0.1
EXPLORATION_MIN         = 0.01
EXPLORATION_DECAY       = 0.999

LEARNING_RATE           = 0.1
DISCOUNT_FACTOR         = 0.6


if __name__=='__main__':
    #create env
    env = gym.make(ENV_NAME)
    #create agent
    agent = QAgent(state_space=env.observation_space, action_space=env.action_space,
                   expl_max=EXPLORATION_MAX, expl_min=EXPLORATION_MIN, expl_decay=EXPLORATION_DECAY,
                   learning_rate=LEARNING_RATE, discount_factor=DISCOUNT_FACTOR)
    #get and parse user args
    args = Parser.parseargs(defaultTrainIterations=10000, defaultEvalIterations=10)
    if args.load:
        agent.load(ENV_NAME, args.loadversion)
    if args.train != 0:
        agent.train(env, args.train)
    if args.eval != 0:
        agent.evaluate(env, args.eval)
    if args.save:
        agent.save(ENV_NAME, args.saveversion)
    if args.render:
        agent.render(env)
    #close env
    env.close()