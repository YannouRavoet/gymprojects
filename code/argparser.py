import argparse

class Parser():
    @staticmethod
    def parseargs(defaultTrainIterations, defaultEvalIterations):
        parser = argparse.ArgumentParser()
        parser.add_argument("-s", "--save", help="save the model after all other actions", action="store_true")
        parser.add_argument("-l", "--load", help="load the model before any other action", action="store_true")
        parser.add_argument("-sv", "--saveversion", help="save to this model version", action="store",default="0.0")
        parser.add_argument("-lv", "--loadversion", help="load this model version", action="store", default="0.0")
        parser.add_argument("-t", "--train", help="number of training iterations", action="store", type=int, default=defaultTrainIterations)
        parser.add_argument("-e", "--eval", help="number of evaluation iterations", action="store", type=int,  default=defaultEvalIterations)
        parser.add_argument("-r", "--render", help="render one episode", action="store_true")
        return parser.parse_args()