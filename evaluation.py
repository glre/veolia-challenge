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
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn import svm
from sklearn.preprocessing import StandardScaler
# End SKLEARN

# Score function
from public_auc_veolia2 import score_function

from tools import *

def learning(M,X,Y):
    Y2 = pd.DataFrame({"target": Y["2014"] + Y["2015"]})
    for i in range(len(Y2)):
        if Y2.iloc[i,0] == 2:
            Y2.iloc[i,0] = 1
    M.fit(X,Y2)

## evaluate the score of a model which was trained
def evaluation(M,Xtest,Ytest):
    pred = M.predict_proba(Xtest)[:,1]
    res = pd.DataFrame({"Id": list(range(len(pred))), "2014": pred, "2015": pred})
    y =roc_auc_score(Ytest["2014"],res.iloc[:,0])
    z =roc_auc_score(Ytest["2015"],res.iloc[:,1])
    return y,z

# test a model: fit + score over training set and test set
def testModel(M,xtrain,ytrain,xtest, ytest):
    scaler = StandardScaler()
    scaler.fit(xtrain)
    xtrain =scaler.transform(xtrain)
    xtest = scaler.transform(xtest)
    learning(M,xtrain,ytrain)
    y,z = evaluation(M,xtest,ytest)
    return y,z


# test a set of models. Each model is tested N times
def testModels(setM, X, Y, N):
    for M in setM:
        print(M)
        print("#########")
        for _ in range(N):
            xtr,xte,ytr,yte = CutData(X,Y,0.4)
            print(testModel(M,xtr,ytr,xte,yte))
