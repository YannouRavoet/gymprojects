from argparser import Parser
import gym
from agent import QAgent

ENV_NAME = "Taxi-v3"

EXPLORATION_MAX         = 0.1
EXPLORATION_MIN         = 0.01
EXPLORATION_DECAY       = 0.999

LEARNING_RATE           = 0.1
DISCOUNT_FACTOR         = 0.6

SAVE_MODEL_EVERY        = 0

def render():
        state = env.reset()
        done = False
        while not done:
            env.render()
            action = agent.model.predict(state)
            new_state, reward, done, info = env.step(action)
            state = new_state
        env.render()

if __name__=='__main__':
    #create env
    env = gym.make(ENV_NAME)
    print(env.unwrapped.spec.id)
    #create agent
    agent = QAgent(obs_space_shp=env.observation_space.n, actions=env.action_space.n,  #both the observation space and action space are discrete
                   expl_max=EXPLORATION_MAX, expl_min=EXPLORATION_MIN, expl_decay=EXPLORATION_DECAY,
                   learning_rate=LEARNING_RATE, discount_factor=DISCOUNT_FACTOR)
    #get and parse user args
    args = Parser.parseargs(defaultTrainIterations=10000, defaultEvalIterations=10)
    if args.load:
        agent.load(env, args.loadversion)
    if args.train != 0:
        agent.train(env, args.train,SAVE_MODEL_EVERY)
    if args.eval != 0:
        print("Evaluation results (lower scores are better):")
        agent.evaluate(env, args.eval)
    if args.save:
        agent.save(env, args.saveversion)
    if args.render:
        agent.render_episode(env, random_action=args.rr)
    #close env
    env.close()