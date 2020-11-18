import pandas as pd
import numpy as np
import math


class reputation:
    def __init__(self, X, recommendations, lambda_, p):
        self.X = X
        self.recommendations = recommendations
        self.lambda_ = lambda_
        self.p = p

    def userReputation(self):
        print("in process")
