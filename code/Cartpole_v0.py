from argparser import Parser
import gym
from wrappers import RewardWrapperPlusMinus, ObservationWrapperReshape
from agent import CartPoleAgent

ENV_NAME = "CartPole-v0"


EXPLORATION_MAX     = 1   #set to 1 for new model
EXPLORATION_MIN     = 0.01
EXPLORATION_DECAY   = 0.9999

LEARNING_RATE       = 0.001
DISCOUNT_FACTOR     = 0.95

MEMORY_SIZE         = 1000000
BATCH_SIZE          = 20

MODEL_SAVE_EVERY    = 10


if __name__== "__main__":
    # create env
    env = gym.make(ENV_NAME)            #gym.make("Cartpole-v0") sets a limit to 200 turns
                                        #gym.make("Cartpole-v0").env doesn't
    env = RewardWrapperPlusMinus(env)
    env = ObservationWrapperReshape(env)

    # create agent
    agent = CartPoleAgent(obs_space_shp=env.observation_space.shape[0], actions=env.action_space.n,  #obs space is Box
                          expl_max=EXPLORATION_MAX, expl_min=EXPLORATION_MIN, expl_decay=EXPLORATION_DECAY,
                          learning_rate=LEARNING_RATE, discount_factor=DISCOUNT_FACTOR,
                          memory_size=MEMORY_SIZE, batch_size=32)
    # get and parse user args
    args = Parser.parseargs(defaultTrainIterations=10000, defaultEvalIterations=10)
    if args.load:
        agent.load(env, args.loadversion)
    if args.train != 0:
        agent.init_fill_memory(env, 20000)
        agent.train(env, args.train, save_i=MODEL_SAVE_EVERY)
    if args.eval != 0:
        print("Evaluation results (higher scores are better):")
        agent.evaluate(env, args.eval)
    if args.save:
        agent.save(env, args.saveversion)
    if args.render:
        agent.render_episode(env, random_action=args.rr)
    # close env
    env.close()