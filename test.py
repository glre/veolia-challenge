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

from sklearn.metrics import roc_auc_score

# Reading Data
In = pd.read_csv("data/inputTrain.csv")
Out = pd.read_csv("data/outputTrain.csv")
Goal = pd.read_csv("data/inputTest.csv")
# End reading



def CutData(In,Out):
    return train_test_split(In, Out, test_size=0.4, random_state=None)

def Featuring(I):
    # pour l'instant, on vire les colonnes catégoriques et on fill les nan avec des 0
    categorical_columns = [d for d in I.columns if I[d].dtype=='object']
    for d in categorical_columns:
        del In[d]
    In.fillna(0,inplace=True)
        


    return (InTrain.values,OutTrain.values,InTest.values,OutTest.values)

def model():
    M = Logit()
    return M

def learn(M,I,O):
    M.fit(I,O)


Featuring(I)

X_train, X_test, y_train, y_test = CutData(In, Out)

