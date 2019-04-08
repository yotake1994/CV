import numpy as np
import scipy as sp
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.externals import joblib
from matplotlib import pyplot as plot





dataset_url = 'http://mlr.cs.umass.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv'
data = pd.read_csv(dataset_url, sep=';')


y = data.quality
x = data.drop('quality', axis=1)

xtrain, xtest, ytrain,ytest = train_test_split(x,y,
                                               test_size=0.2,
                                               random_state=123,
                                               stratify=y)

scaler= preprocessing.StandardScaler().fit(xtrain)

xtrain_scaled= scaler.transform(xtrain)

xtest_scaled= scaler.transform(xtest)

pipeline = make_pipeline(preprocessing.StandardScaler(),
                         RandomForestRegressor(n_estimators=100))

hyperparameters = {'randomforestregressor__max_features' : ['auto', 'sqrt', 'log2'],
                   'randomforestregressor__max_depth': [None, 5,3,1]}

clf= GridSearchCV(pipeline, hyperparameters, cv=10)
clf.fit(xtrain,ytrain)

ypred=clf.predict(xtest)

print(r2_score(ytest, ypred))


joblib.dump(clf, 'rfregressor.pkl')
clf2= joblib.load('rfregressor.pkl')

print(clf2.predict(xtest))









