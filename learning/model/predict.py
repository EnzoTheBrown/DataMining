import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPRegressor


class Prediction:
    def __init__(self, df):
        self.df = df:
        self.clusters = {}
    
    # A cluster is a list of square 
    def get_cluster(self, cluster):
        self.clusters[cluster] = self.df[self.df['Square'] in cluster]

    def mpl_prediction(self, cluster, datasize):
        x = self.clusters[cluster]
        n = len(x)
        y = x['overload']



    
    

