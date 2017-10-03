import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time
import math

import random
import itertools as it

# SKLEARN
import sklearn as sk
import scikitplot as skplt
from sklearn.linear_model import LogisticRegression as Logit
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import GradientBoostingClassifier as GradB
from sklearn.ensemble import RandomForestClassifier as Forest
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn import svm
from sklearn.preprocessing import StandardScaler
# End SKLEARN

# My files
from reading import *
from featuring import *
from tools import *
from evaluation import *


def prediction(M, X, Y, Goal,name):
    X2 = X.copy()
    G = featuringInput(Goal)
    scaler = StandardScaler()
    scaler.fit(X2)
    X2 =scaler.transform(X2)
    G = scaler.transform(G)
    learning(M,X2,Y)
    print(evaluation(M,X2,Y))
    prediction = M.predict_proba(G)[:,1]
    resultat = pd.DataFrame({"Id": Goal["Id"], "2014": prediction, "2015": prediction})
    resultat.to_csv("result/" + (name + ".csv"), sep=";", columns=["Id", "2014", "2015"], index=False)


R2 = 3
R = Reading() # Reading data
X = featuringInput(R.train) #Featuring In
Goal = R.test
Y = R.output
xtrain,xtest,ytrain,ytest = CutData(X,Y,0.4)
M = Logit()
#learning(M,xtrain,ytrain)
#pred = M.predict_proba(xtest)[:,0]
S = set()
#M = Logit()
#M2 = GradB()
MLP = MLPClassifier(activation='logistic', max_iter=6000000, learning_rate_init=0.0001, tol = 1e-10)
# MLP2 semble foncionner super bien !
MLP2 = MLPClassifier(activation='logistic', max_iter=6000000, learning_rate_init=0.0001, tol = 1e-10 , hidden_layer_sizes=[10,10])
S.add(MLP2)
clf = svm.SVC(probability=True)
#testModels(S,X,Y,5)
#prediction(MLP,X,Y,Goal,"hahaha")

#MLP = MLPClassifier(activation='logistic', max_iter=6000000, learning_rate_init=0.0001, tol = 1e-10 , hidden_layer_sizes=[10])
#>>> prediction(MLP,X,Y,Goal,"haha")
#/home/leon/.local/lib/python3.5/site-packages/sklearn/neural_network/multilayer_perceptron.py:912: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shap#e of y to (n_samples, ), for example using ravel().
#  y = column_or_1d(y, warn=True)
#(0.90116592749278845, 0.87886204925916123)
#>>> MLP = MLPClassifier(activation='logistic', max_iter=6000000, learning_rate_init=0.0001, tol = 1e-10 , hidden_layer_sizes=[10,10])
#>>> prediction(MLP,X,Y,Goal,"haha")
#(0.90492899450927233, 0.88031445576571932)
#>>> MLP = MLPClassifier(activation='logistic', max_iter=6000000, learning_rate_init=0.0001, tol = 1e-10 , hidden_layer_sizes=[40,40])
#>>> prediction(MLP,X,Y,Goal,"haha")
#(0.89831733250748425, 0.87901258659381409)
#>>> MLP = MLPClassifier(activation='logistic', max_iter=6000000, learning_rate_init=0.0001, tol = 1e-10 , hidden_layer_sizes=[40,40])



