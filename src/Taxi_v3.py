from argparser import Parser
import gym
from models import QTable
from agent import QAgent

ENV_NAME = "Taxi-v3"

EXPLORATION_MAX         = 0.1
EXPLORATION_MIN         = 0.01
EXPLORATION_DECAY       = 0.999

LEARNING_RATE           = 0.1
DISCOUNT_FACTOR         = 0.6

SAVE_MODEL_EVERY        = 0

if __name__=='__main__':
    #create env
    env = gym.make(ENV_NAME)
    print(env.unwrapped.spec.id)
    #create agent
    model = QTable(nostates=env.observation_space.n, noactions=env.action_space.n,
                            learning_rate=LEARNING_RATE, discount_factor=DISCOUNT_FACTOR)
    agent = QAgent(actions=env.action_space.n,
                   expl_max=EXPLORATION_MAX, expl_min=EXPLORATION_MIN, expl_decay=EXPLORATION_DECAY,
                   model=model)

    #get and parse user args
    args = Parser.parseargs(defaultTrainIterations=10000, defaultEvalIterations=10)
    if args.load:
        agent.load(env, args.loadversion)
    if args.train != 0:
        agent.train(env, iterations=args.train,train_s=1, save_i=SAVE_MODEL_EVERY)
    if args.eval != 0:
        print("Evaluation results (lower scores are better):")
        agent.evaluate(env, args.eval)
    if args.save:
        agent.save(env, args.saveversion)
    #close env
    env.close()