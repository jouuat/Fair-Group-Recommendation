import argparse
import sys

from DATParser import DATParser
from ValidateConfig import ValidateConfig
from dataset import dataset
from groupDetection import groupDetection
from recommender import recommendationModel
from groupModeling import groupModeling


def run():
    try:
        # des del terminal agafara l'argument com  a configFile
        argp = argparse.ArgumentParser(description='Group Recommendations')
        argp.add_argument('configFile', help='configuration file path')
        args = argp.parse_args()

        print ('-----------------------------------------------------------------------')
        print ('-------------------------Group Recommendations-------------------------')
        print ('-----------------------------------------------------------------------\n')
        config = DATParser.parse(args.configFile)
        ValidateConfig.validate(config)
        data = dataset(config)
        ratings, movies = data.getData()
        # recommender model
        recommenderModel = recommendationModel(config, ratings, movies)
        tfRecommender = recommenderModel.train()
        groups = groupDetection(config, ratings, movies)
        groups = groups.detect()
        modeling = groupModeling(config, groups)
        modeling.model(tfRecommender)

    except Exception as e:
        print
        print ('Exception:', e)
        import traceback
        traceback.print_exc(file=sys.stdout)
        print
        return(1)


if __name__ == '__main__':
    sys.exit(run())
