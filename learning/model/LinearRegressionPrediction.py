import numpy as np
import pandas as pd
from sklearn import linear_model
from sklearn.model_selection import train_test_split


def linear_regression_prediction(df):
    dim = df[['SMSin', 'SMSout', 'Callin', 'Callout', 'Internet', 'overload']]
    X_train, X_test, y_train, y_test = train_test_split(dim[['SMSin', 'SMSout', 'Callin', 'Callout', 'Internet']], dim['overload'], test_size=.4, random_state=0)
    regr = linear_model.LinearRegression()
    regr.fit(X_train, y_train)
    y_hat = regr.predict(X_test)

    print(list(map(int,y_test)), y_hat)


    


