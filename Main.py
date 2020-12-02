import matplotlib.pyplot as plt
import warnings
import argparse
import sys
import numpy as np


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
        # ALL = TRUE
        if config.all:
            config.listOfUsersPerGroup = list(config.listOfUsersPerGroup)  # DATParser returns a map object and we want a list
            config.listOfGroupsModeling = list(config.listOfGroupsModeling)
            scores = {'x': config.listOfUsersPerGroup}
            pearsons = list()
            for modelingStrategy in config.listOfGroupsModeling:
                scores[modelingStrategy] = list()
            for usersPerGroup in config.listOfUsersPerGroup:
                config.usersPerGroup = usersPerGroup
                groupsClass = groupDetection(config, ratings_pd)
                groups, pearson = groupsClass.detect()
                pearsons.append(0.01 * pearson)
                for modelingStrategy in config.listOfGroupsModeling:
                    print ('generating recommnedations with', modelingStrategy, 'technique for groups with', usersPerGroup, ' users \n')
                    config.groupModeling = modelingStrategy
                    modeling = groupModeling(config, groups, ratings_pd)
                    score = modeling.model(tfRecommender)
                    scores[modelingStrategy].append(score)
            # multiple line plot
            palette = plt.get_cmap('Set1')  # create a color palette
            color = 0
            for modelingStrategy in config.listOfGroupsModeling:
                color += 1
                print('x', scores['x'], 'modelingStrategy', scores[modelingStrategy], 'color', palette(color))
                plt.plot(scores['x'], scores[modelingStrategy], marker='o', markerfacecolor=palette(color), markersize=3, color=palette(color), linewidth=1, label=modelingStrategy)
                plt.fill_between(scores['x'], np.array(scores[modelingStrategy]) - np.array(pearsons), np.array(scores[modelingStrategy]) + np.array(pearsons), alpha=0.5, facecolor=palette(color))
            plt.title('%s Vs users per group for %s %s groups' % (config.metric, config.numOfGroups, config.groupDetection))
            plt.xlabel('users per group')
            plt.ylabel('%s' % (config.metric))
            plt.legend()
            plt.show()

        # INDIVIDUAL CASES
        else:
            print ('----------------------------Creating groups----------------------------\n')
            groupsClass = groupDetection(config, ratings_pd)
            groups, pearson = groupsClass.detect()
            groupsClass.groupInfo()
            print ('-----------------generating recommnedations for each group-------------\n')
            modeling = groupModeling(config, groups, ratings_pd)
            score = modeling.model(tfRecommender)

    except Exception as e:
        print
        print ('Exception:', e)
        import traceback
        traceback.print_exc(file=sys.stdout)
        print
        return(1)


if __name__ == '__main__':
    sys.exit(run())
