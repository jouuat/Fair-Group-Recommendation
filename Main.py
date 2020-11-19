import warnings
import argparse
import sys

from DATParser import DATParser
from ValidateConfig import ValidateConfig
from dataset import dataset
from groupDetection import groupDetection
from recommender import recommendationModel
from groupModeling import groupModeling

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # don't show info, warnings and errors of tensorflow

warnings.filterwarnings("ignore")  # don't show warnings (pearsonr)


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
        print ('----------------------------Loading dataset----------------------------\n')
        data = dataset(config)
        ratings_tf, movies_tf, ratings_pd = data.getData()
        print ('--------------------Trainning the Recommender model--------------------\n')
        recommenderModel = recommendationModel(config, ratings_tf, movies_tf, ratings_pd)
        tfRecommender = recommenderModel.train()
        print ('----------------------------Creating groups----------------------------\n')
        groupsClass = groupDetection(config, ratings_pd)
        groups = groupsClass.detect()
        groupsClass.groupInfo()
        print ('-----------------generating recommnedations for each group-------------\n')
        modeling = groupModeling(config, groups, ratings_pd)
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
