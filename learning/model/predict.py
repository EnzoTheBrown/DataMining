import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPRegressor


class PredictionResult:
    def __init__(self, df):
        self.df = df
        self.clusters = {}
    
    def dict2clusters(self, dict_):
        clusters = set(dict_.values())
        temp = {}
        for cluster in clusters:
            temp[cluster] = []
        for k,v in dict_.items():
            temp[v].append(k)
        return temp   

    def reset_clusters(self):
        self.clusters = {}

    # A cluster is a list of square 
    def get_cluster(self, name, cluster):
        self.clusters[name] = self.df[self.df['Square'].isin(cluster)]

    def predict(self, cluster):
        print(self.clusters)
        data = self.clusters[cluster]
        return {'time':list(data['Time']), 'value':list(data['SMSin'])}


        


    
    

