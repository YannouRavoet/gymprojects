from argparser import Parser
import gym
from wrappers import RewardSign, ObservationCHW1x84x84, FrameStacker, FireOnDeath
from agent import DoubleDQN, DQNAgent
from models import AtariNetwork

ENV_NAME = "Breakout"+"Deterministic-v4"    #Deterministic: skip 4 frames at each step
                                            #v0 25% chance of doing last action instead of new action
                                            #v4 always issue the action that is passed



EXPLORATION_MAX = 1            #set to 1 for new model
EXPLORATION_MIN = 0.005
EXPLORATION_DECAY = 0.999       #higher is slower decay

LEARNING_RATE   = 0.1
DISCOUNT_FACTOR = 0.99


MEMORY_SIZE = 1000000
BATCH_SIZE  = 32

MODEL_SAVE_EVERY = 5

if __name__== "__main__":
    # create env
    env = gym.make(ENV_NAME)
    env = RewardSign(env)
    env = ObservationCHW1x84x84(env)
    env = FrameStacker(env, 4)
    if 'FIRE' in env.get_action_meanings():
        env = FireOnDeath(env, env.get_action_meanings().index('FIRE'))
    # create agent
    model = AtariNetwork(learning_rate=LEARNING_RATE, discount_factor=DISCOUNT_FACTOR,
                         input_shape=env.observation_space.shape, output_shape=env.action_space.n)
    target_model = AtariNetwork(learning_rate=LEARNING_RATE, discount_factor=DISCOUNT_FACTOR,
                         input_shape=env.observation_space.shape, output_shape=env.action_space.n)
    agent = DoubleDQN(actions=env.action_space.n,
                      expl_max=EXPLORATION_MAX, expl_min=EXPLORATION_MIN, expl_decay=EXPLORATION_DECAY,
                      model =model, target_model=target_model,
                      memory_size=MEMORY_SIZE, batch_size=32)
    # get and parse user args
    args = Parser.parseargs(defaultTrainIterations=10000, defaultEvalIterations=10)
    if args.load:
        agent.load(env, args.loadversion)
    if args.train != 0:
        agent.init_fill_memory(env, 20000)
        agent.train(env, args.train, train_s=4, save_i=MODEL_SAVE_EVERY)
    if args.eval != 0:
        print("Evaluation results (higher scores are better):")
        agent.evaluate(env, args.eval)
    if args.save:
        agent.save(env, args.saveversion)
    if args.render:
        agent.render_episode(env, random_action=args.renderrandom)
    # close env
    env.close()