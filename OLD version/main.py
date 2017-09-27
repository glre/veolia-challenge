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
# End SKLEARN

# Score function
from public_auc_veolia2 import score_function

from featuring import *
from tools import *
from evaluation import *
from prediction import *

# Reading Data. In, Out and Goal do not have the "Id" column.
In = pd.read_csv("data/inputTrain.csv")
Out = pd.read_csv("data/outputTrain.csv")
Goal = pd.read_csv("data/inputTest.csv")
GoalTemp = GoalTemp.iloc[:,1:]
# End reading

# N is the number of tests made for each model
def testModels(setM, X, Y, N):
    ltests = []
    for _ in range(N):
        xtr,ytr,xte,yte = cutData(X,Y)
        l.append([xtr,ytr,xte,yte])
    for M in setM:
        print(M)
        res2014 = []
        res2015 = []
        for D in ltests:
            M.fit(D[0],D[1])
            x,y = testModel(M,D[2],D[3])
            

def resultat(nameOutput):
    bestFeaturing(In)
    bestFeaturing(Goal)
    X_train, X_test, y_train, y_test = CutData(In, Out,0)
    M = Logit()
    M.fit(X_train, y_train)
    writeOutput(M,nameOutput)

def launch(N,ratio):
    for i in range(N):
        X_train, X_test, y_train, y_test = CutData(In, Out,ratio)
        M = Logit()
        M.fit(X_train, y_train)
        y,z = testModel(M,X_test,y_test)
        print(y)
        print(z)
        y,z = testModel(M,In,Out)
        print(y)
        print(z)
        var = input("we take it dude?(y/q : Yes, Quit)")
        if var == "y":
            var2 = input("name of the output?")
            writeOutput(M,var2)
        if var == "q":
            break
        print("##################")
#pred = M.predict_proba(X_test)[:,1]
#res = pd.DataFrame({"Id": list(range(len(pred))), "2014": pred, "2015": pred})
#res.to_csv("test.csv", sep=";", columns=["Id", "2014", "2015"], index=False)

