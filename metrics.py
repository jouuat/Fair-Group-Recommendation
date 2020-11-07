import pandas as pd
import numpy as np


class metrics:
    def __init__(self, X):
        self.X = X

    def z_recall(self):
        # average
        for id in list(self.X.columns):
            self.X[id] = (self.X[id] >= 4).astype(int)
        print("working on it")
