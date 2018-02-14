from mining.src.model.clustering import apply_kmeans, apply_dbscan
from init_data import init_data, apply_overload
from mining.src.model.mining import *
from sklearn import metrics

import numpy as np
import pandas as pd
from sklearn import preprocessing, cross_validation, svm
from sklearn.linear_model import LinearRegression
import math
import matplotlib.pyplot as plt


cluster = 4
df = apply_overload(init_data(10))
df = df[['Time', 'Callin', 'Callout', 'SMSin', 'SMSout', 'Internet']]

def predict(df, forecast_col='SMSin'):
	d = df[['Time', 'SMSin']]
	d = d.groupby(['Time'])['SMSin'].mean()
	d = d.reset_index()
	d.fillna(value=-99999, inplace=True)
	forecast_out = int(math.ceil(0.01 * len(d)))
	d['label'] = d[forecast_col].shift(-forecast_out)
	d.dropna(inplace=True)
	X = np.array(d.drop(['label'], 1))
	y = np.array(d['label'])

	X = preprocessing.scale(X)
	X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y, test_size=0.2)

	clfl = LinearRegression()
	clfl.fit(X_train, y_train)
	lm = clfl.predict(X_test)
	confidencel = clfl.score(X_test, y_test)

	clf = svm.SVR()
	clf.fit(X_train, y_train)
	svr = clf.predict(X_test)
	confidencePourConfidence = clf.score(X_test, y_test)
	
	return lm, confidencel, svr, confidencePourConfidence, y_test
	
	
def plotMeIfYouCan(df):
	lm, confidencel, svr, confidence, y_test = predict(df)
	plt.plot(lm, color='red', label='LinearRegression: confidence = ' + str(confidencel) )
	plt.plot(svr, color='blue', label='SVM: confidence = ' + str(confidence))
	plt.plot(y_test, color='green', label='truth')
	plt.legend()

	plt.show()
	
	
# loading data:
df = apply_overload(init_data(10))

# clustering with kmeans
clustering = ClusteringResult(5, apply_overload(init_data(10)))
clusters = clustering.get_kmeans_result(cluster)['labels']

# affect data to clusters
def chose(x, clusters):
	if str(x) in clusters:
		return clusters[str(x)]
	return '0'
		
def set_cluster(df, clusters):
	df['#Cluster'] = df['Square'].apply(lambda x: chose(x, clusters))

set_cluster(df, clusters)

plotMeIfYouCan(df)
for i in set(df['#Cluster']):
	plotMeIfYouCan(df[df['#Cluster'] == i])


