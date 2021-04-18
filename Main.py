import matplotlib.pyplot as plt
import warnings
import argparse
import sys
import numpy as np
import os
import json
from os import path  # or import os.path
import seaborn as sns
import pandas as pd
from scipy.stats import f_oneway

from DATParser import DATParser
from ValidateConfig import ValidateConfig
from dataset import dataset
from groupDetection import groupDetection
from recommender import recommendationModel
from groupModeling import groupModeling
from metrics import metrics

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # don't show info, warnings and errors of tensorflow
warnings.filterwarnings("ignore")  # don't show warnings (pearsonr)
sns.set_theme(style="whitegrid")
# cwd = os.getcwd()
# print(cwd)


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
        checkpoint_filepath = '/tmp/checkpoint_' + str(config.dataset)
        print ('----------------------------Loading dataset----------------------------\n')
        data = dataset(config)
        ratings_tf, candidates_tf, ratings_pd = data.getData()
        sizeDataset = len(ratings_pd["user_rating"].tolist())
        train = ratings_pd.head(int(sizeDataset * 0.8))
        test = ratings_pd.tail(int(sizeDataset * 0.2))
        print ('--------------------Trainning the Recommender model--------------------\n')
        recommenderModel = recommendationModel(config, ratings_tf, candidates_tf, ratings_pd, checkpoint_filepath)
        tfRecommender = recommenderModel.train()
        config.listOfUsersPerGroup = list(config.listOfUsersPerGroup)  # DATParser returns a map object and we want a list
        config.listOfGroupsModeling = list(config.listOfGroupsModeling)
        scores = {'x': config.listOfUsersPerGroup}
        groupScores_pd_first = True
        # pearsons = list()
        for modelingStrategy in config.listOfGroupsModeling:
            scores[modelingStrategy] = list()
        for usersPerGroup in config.listOfUsersPerGroup:
            print ('------------------', usersPerGroup, ' users per group--------------------\n')
            config.usersPerGroup = int(usersPerGroup)
            group_data_path = 'groups_' + str(config.dataset) + '_' + str(config.groupDetection) + '_' + str(config.usersPerGroup) + '.txt'
            if path.exists(group_data_path):
                with open(group_data_path, "r") as read_file:
                    groups = json.load(read_file)
            else:
                groupsClass = groupDetection(config, ratings_pd, group_data_path)
                groups = groupsClass.detect()
            # pearsons.append(0.01 * pearson)
            for modelingStrategy in config.listOfGroupsModeling:
                config.groupModeling = str(modelingStrategy)
                recommendation_data_path = 'recommendations_' + \
                    str(config.dataset) + '_' + str(config.groupDetection) + '_' + str(modelingStrategy) + '_' + str(config.usersPerGroup) + '.txt'
                if path.exists(recommendation_data_path):
                    print (modelingStrategy, 'group modeling technique \n')
                    with open(recommendation_data_path, "r") as read_file:
                        recommendations = json.load(read_file)
                else:
                    print (modelingStrategy, 'group modeling technique \n')
                    modeling = groupModeling(config, groups, train, recommendation_data_path)
                    recommendations = modeling.model(tfRecommender)
                print ('Computing score \n')
                metrics_ = metrics(config, test, groups, recommendations)
                score, groupScores = metrics_.getScore()
                if modelingStrategy == "gfar" or modelingStrategy == "arm":
                    print(groupScores)
                print (modelingStrategy, config.metric, ':', score, '\n')
                if config.metric.lower() != "fndcg":
                    scores[modelingStrategy].append(score)
                    # create boxplot dataset & x, y labels
                    usersPerGroup_ = [usersPerGroup] * len(groupScores)
                    technique_ = [modelingStrategy] * len(groupScores)
                if groupScores_pd_first:
                    if config.metric.lower() == "fndcg":
                        groupScores_pd_first = False
                        groupScores_pd = pd.DataFrame(data=[[score, groupScores, modelingStrategy]], columns=["accuracy", "fairness", "technique"])
                    else:
                        groupScores_pd_first = False
                        groupScores_pd = pd.DataFrame(data=list(zip(groupScores, usersPerGroup_, technique_)), columns=["groupScores", "usersPerGroup", "technique"])
                else:
                    if config.metric.lower() == "fndcg":
                        groupScores_pd2 = pd.DataFrame(data=[[score, groupScores, modelingStrategy]], columns=["accuracy", "fairness", "technique"])
                        groupScores_pd = pd.concat([groupScores_pd, groupScores_pd2])
                    else:
                        groupScores_pd2 = pd.DataFrame(data=list(zip(groupScores, usersPerGroup_, technique_)), columns=["groupScores", "usersPerGroup", "technique"])
                        groupScores_pd = pd.concat([groupScores_pd, groupScores_pd2])
            # show the different boxplots
            # temp_boxplot = groupScores_pd[groupScores_pd["usersPerGroup"] == usersPerGroup]
            # sns.boxplot(x=temp_boxplot["technique"], y=temp_boxplot["groupScores"])
            # plt.show()
            # plt.pause(0.001)
        # check if our implementation is statistically different
        '''techniques_ = config.listOfGroupsModeling
        otherTechniques = techniques_.remove("our2")
        our = groupScores_pd[groupScores_pd["technique"] == "our2"]
        print(otherTechniques)
        for modelingStrategy in otherTechniques:
            other = groupScores_pd[groupScores_pd["technique"] == modelingStrategy]
            probability = []
            for usersPerGroup in config.listOfUsersPerGroup:
                our_ = our[our["usersPerGroupsers"] == usersPerGroup]
                our_ = list(our_["technique"].values)
                other_ = other[other["usersPerGroupsers"] == usersPerGroup]
                other_ = list(other_["technique"].values)
                stat, p = f_oneway(our_, other_)
                probability.append(p)
            p = sum(probability) / len(probability)
            if p > 0.05:
                print('Probably our technique and', modelingStrategy, 'have the same distribution')
            else:
                print('Probably our technique and', modelingStrategy, 'have different distribution')'''
        # print the mean scores for each modelling technique
        if config.metric.lower() == "fndcg":
            fndcg_lambda = 0.5
            for modelingStrategy in config.listOfGroupsModeling:
                strategyScores = groupScores_pd[groupScores_pd["technique"] == modelingStrategy]
                accuracy = strategyScores["accuracy"]
                fairness = strategyScores["accuracy"]
                accuracy = accuracy.pow(2)
                fairness = fairness.pow(2)
                accuracy = accuracy.mul(fndcg_lambda)
                fairness = fairness.mul((1 - fndcg_lambda))
                sum_ = accuracy.add(fairness, axis=0)
                sum_root = sum_.pow(0.5)
                total = sum_root.sum(axis=0) / len(config.listOfGroupsModeling)
                print(config.metric, 'mean score with', modelingStrategy, 'modeling strategy is:', total)
        if config.metric.lower() != "fndcg":
            for modelingStrategy in config.listOfGroupsModeling:
                meanScore = sum(scores[modelingStrategy]) / len(scores[modelingStrategy])
                print(config.metric, 'mean score with', modelingStrategy, 'modeling strategy is:', meanScore)
        # multiple line plot
        print(groupScores_pd)
        palette = plt.get_cmap('Set1')  # create a color palette
        color = 0
        if config.metric.lower() == "fndcg":
            for modelingStrategy in config.listOfGroupsModeling:
                sns.scatterplot(data=groupScores_pd, x="fairness", y="accuracy", hue="technique")
            # plt.title('Accuracy (%s) Vs Fairness / %s' % (config.metric, config.groupDetection))
            # plt.xlabel('Fairness')
            # plt.ylabel('Accuracy(%s)' % (config.metric))
        else:
            for modelingStrategy in config.listOfGroupsModeling:
                color += 1
                plt.plot(scores['x'], scores[modelingStrategy], marker='o', markerfacecolor=palette(color), markersize=3, color=palette(color), linewidth=1, label=modelingStrategy)
                # plt.fill_between(scores['x'], np.array(scores[modelingStrategy]) - np.array(pearsons), np.array(scores[modelingStrategy]) + np.array(pearsons), alpha=0.5, facecolor=palette(color))
            plt.title('%s Vs users per %s group' % (config.metric, config.groupDetection))
            plt.xlabel('users per group')
            plt.ylabel('%s' % (config.metric))
            plt.legend()
        plt.show()
        # plt.pause(0.001)

    except Exception as e:
        print
        print ('Exception:', e)
        import traceback
        traceback.print_exc(file=sys.stdout)
        print
        return(1)


if __name__ == '__main__':
    sys.exit(run())
