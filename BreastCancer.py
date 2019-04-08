import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.externals import joblib
from matplotlib import pyplot as plt

data = pd.read_csv('data.csv')
y= data.diagnosis
x=data.drop('diagnosis', axis=1)
xtrain, xtest, ytrain, ytest= train_test_split(x,y,
                                               test_size=0.1,
                                               random_state=123,
                                               stratify=y)
scaler= preprocessing.StandardScaler().fit(xtrain)
xtrain_scaled= scaler.transform(xtrain)
xtest_scaled = scaler.transform(xtest)

pipeline = make_pipeline(preprocessing.StandardScaler(),
                         RandomForestRegressor(n_estimators=100))
hyperparamaters = {'randomforestregressor__max_features': ['auto', 'sqrt', 'log2'],
                   'randomforestregressor_max_depth': [None,5,3,1]}
clf = GridSearchCV(pipeline, hyperparamaters, cv=10)

clf.fit(xtrain,ytrain)