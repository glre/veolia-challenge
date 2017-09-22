import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time
import math

import sklearn as sk
import random

# MODELS
from sklearn.linear_model import LogisticRegression as Logit
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split

from public_auc_veolia2 import score_function

# Reading Data. In et Out n'ont pas les Id
InTemp = pd.read_csv("data/inputTrain.csv")
In = InTemp.iloc[:,1:]
OutTemp = pd.read_csv("data/outputTrain.csv")
Out = OutTemp.iloc[:,1:]
GoalTemp = pd.read_csv("data/inputTest.csv")
Goal = GoalTemp[:,1:]
# End reading


# Cut data into training set and test set, to do cross validation
# Return a training set, a test set
def CutData(In,Out):    
    indexOnes = [i for i in range(len(Out)) if (Out.iloc[i]["2014"] == 1 or Out.iloc[i]["2015"] == 1)]
    indexNotOnes = list(set(range(len(Out))) - set(indexOnes))

    InOnes = In.iloc[indexOnes]
    OutOnes = Out.iloc[indexOnes]
    InNotOnes = In.iloc[indexNotOnes]
    OutNotOnes = Out.iloc[indexNotOnes]
    
    XOnes_train, XOnes_test, yOnes_train, yOnes_test = train_test_split(InOnes, OutOnes, test_size=0.4, random_state=None)
    XNotOnes_train, XNotOnes_test, yNotOnes_train, yNotOnes_test = train_test_split(InNotOnes, OutNotOnes, test_size=0.4, random_state=None)

    del yOnes_train["2014"]
    yOnes_train["2015"] = 1
    del yNotOnes_train["2014"]

    
    X_train = pd.concat([XOnes_train,XNotOnes_train])
    X_test = pd.concat([XOnes_test,XNotOnes_test])

    y_train = pd.concat([yOnes_train,yNotOnes_train])
    y_test = pd.concat([yOnes_test,yNotOnes_test])

    return (X_train, X_test, y_train, y_test)


def Featuring(I):
    # pour l'instant, on vire les colonnes catégoriques et on fill les nan avec des 0
    categorical_columns = [d for d in I.columns if I[d].dtype=='object']
    for d in categorical_columns:
        del In[d]
    I.fillna(0,inplace=True)
    
def model():
    M = Logit()
    return M

def learn(M,I,O):
    M.fit(I,O)


Featuring(In)

X_train, X_test, y_train, y_test = CutData(In, Out)

M = Logit()
M.fit(X_train, y_train)
pred = M.predict_proba(X_test)[:,1]
res = pd.DataFrame({"Id": list(range(len(pred))), "2014": pred, "2015": pred})
res.to_csv("test.csv", sep=";", columns=["Id", "2014", "2015"], index=False)

