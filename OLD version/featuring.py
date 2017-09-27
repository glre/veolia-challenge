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
import pandas as pd

# Given a dataframe df, and a column name cN whose objects are categorical, vectorialize them 
def vectorialization(df,colName):
    s = set() # The set of all kind of object we can find
    for i in range(len(df)):
        s.add(df[colName].iloc[i])
    l = list(s)
    for i in range(len(s)):
        df[colName + str(i)] = 0
    for i in range(len(df)):
        index = l.index(df[colName].iloc[i])
        df.iloc[i,-index-1] = 1
    del df[colName]


def featuringInput(In):
    X = In.copy()
    del X["Id"]
    del X["Feature4"]
    categorical_columns = [d for d in X.columns if X[d].dtype=='object']
    for d in categorical_columns:
        vectorialization(X,d)
    X.fillna(0,inplace=True)
    return X


def featuringOutput(Out):
    Y = pd.DataFrame({"target": Out["2014"] + Out["2015"]})
    for i in range(len(Y)):
        if Y["target"][i] == 2:
            Y["target"][i] = 1
    return Y

def FeaturingInputSimple(In):
    # For now, don't use categorical features + nan -> 0
    X = In.copy()
    categorical_columns = [d for d in X.columns if X[d].dtype=='object']
    for d in categorical_columns:
        del X[d]
    X.fillna(0,inplace=True)


# test all subsets of features
def testFeatures(M,I,i,N):
    nb = len(I.columns)
    subsets = list(map(set,list(findsubsets(I.columns,nb-i))))
    for l in subsets:
        print("###############")
        print("###############")
        print("features present:")
        print(set(I.columns) - set(l))
        print("features missing:")
        print(l)
        print()
        I2 = I.copy()
        for e in l:
            del I2[e]
        Featuring(I2)
        Test2014 = []
        Test2015 = []
        Total2014 = []
        Total2015 = []
        for _ in range(N):
            print("test numéro "+ str(_))
#            print()
            X_train, X_test, y_train, y_test = CutData(I2, Out,0.4)
            M = Logit()
            M.fit(X_train, y_train)
            y,z = testModel(M,X_test,y_test)
            Test2014.append(y)
            Test2015.append(z)
            y,z = testModel(M,I2,Out)        
            Total2014.append(y)
            Total2015.append(z)
            print("Done")
#        print(Test2014) # Pour plus de détails
        print(np.mean(Test2014), np.std(Test2014))
#        print(Test2015) # Pour plus de détails
        print(np.mean(Test2015), np.std(Test2015))
        
#        print(Total2014) # Pour plus de détails
        print(np.mean(Total2014), np.std(Total2014))
#        print(Test2015) # Pour plus de détails
        print(np.mean(Total2015), np.std(Total2015))


# Some tools
def powerset(iterable):
    s = list(iterable)
    return it.chain.from_iterable(it.combinations(s,r) for r in range(len(s)+1))


def findsubsets(S,m):
    return set(it.combinations(S, m))
