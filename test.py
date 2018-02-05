import pandas as pd
from learning.model.LinearRegressionPrediction import linear_regression_prediction
import matplotlib.pyplot as plt


df = pd.read_csv('data/sms-call-internet-mi-2014-01-01.txt-sample.csv')
print(df)
df.columns = ['Id', 'Square', 'Time', 'Country', 'SMSin', 'SMSout', 'Callin', 'Callout', 'Internet']
a, b = linear_regression_prediction(df)

