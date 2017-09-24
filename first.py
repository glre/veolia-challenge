import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time
import math

import sklearn as sk
import scikitplot as skplt
import random

import itertools as it

# MODELS
from sklearn.linear_model import LogisticRegression as Logit
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
# End Models

# Score functione
from public_auc_veolia2 import score_function
from sklearn.metrics import roc_auc_score

# Reading Data. In, Out and Goal do not have the "Id" column.
InTemp = pd.read_csv("data/inputTrain.csv")
In = InTemp.iloc[:,1:]
OutTemp = pd.read_csv("data/outputTrain.csv")
Out = OutTemp.iloc[:,1:]
GoalTemp = pd.read_csv("data/inputTest.csv")
Goal = GoalTemp.iloc[:,1:]
# End reading


# Some tools
def powerset(iterable):
    s = list(iterable)
    return it.chain.from_iterable(it.combinations(s,r) for r in range(len(s)+1))


def findsubsets(S,m):
    return set(it.combinations(S, m))

# Cut data into training set and test set, to do cross validation
# Return a training set, a test set
def CutData(In,Out,ratio):    
    indexOnes = [i for i in range(len(Out)) if (Out.iloc[i]["2014"] == 1 or Out.iloc[i]["2015"] == 1)]
    indexNotOnes = list(set(range(len(Out))) - set(indexOnes))

    InOnes = In.iloc[indexOnes]
    OutOnes = Out.iloc[indexOnes]
    InNotOnes = In.iloc[indexNotOnes]
    OutNotOnes = Out.iloc[indexNotOnes]
    
    XOnes_train, XOnes_test, yOnes_train, yOnes_test = train_test_split(InOnes, OutOnes, test_size=ratio, random_state=None)
    XNotOnes_train, XNotOnes_test, yNotOnes_train, yNotOnes_test = train_test_split(InNotOnes, OutNotOnes, test_size=ratio, random_state=None)

    del yOnes_train["2014"]
    yOnes_train["2015"] = 1
    del yNotOnes_train["2014"]
    
    X_train = pd.concat([XOnes_train,XNotOnes_train])
    X_test = pd.concat([XOnes_test,XNotOnes_test])

    y_train = pd.concat([yOnes_train,yNotOnes_train])
    y_test = pd.concat([yOnes_test,yNotOnes_test])

    return (X_train, X_test, y_train, y_test)


def FeaturingSimple(I):
    # For now, don't use categorical features + nan -> 0
    categorical_columns = [d for d in I.columns if I[d].dtype=='object']
    for d in categorical_columns:
        del I[d]
    I.fillna(0,inplace=True)

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
        

def Featuring(I):
#    del I["Feature4"]
    categorical_columns = [d for d in I.columns if I[d].dtype=='object']
    for d in categorical_columns:
        vectorialization(I,d)
    I.fillna(0,inplace=True)
    ## On catégorise l'année de construction, peut-être qu'il faudra le virer
    #vectorialization(I, "YearConstruction")

# test all i-subsets of features over the model M
# repeat N tests over all possible subset
def testFeatures(M,I,i,N):
    nb = len(I.columns)
    subsets = list(map(set,list(findsubsets(I.columns,nb-i))))
    for l in subsets:
        print("###############")
        print("###############")
        print(l)
        print()
        I2 = I.copy()
        for e in l:
            del I2[e]
        Featuring(I2)
        for _ in range(N):
            print("test numéro "+ str(_))
            print()
            X_train, X_test, y_train, y_test = CutData(I2, Out,0.4)
            M = Logit()
            M.fit(X_train, y_train)
            testModel(M,X_test,y_test)
            testModel(M,I2,Out)        
    


def model():
    M = Logit()
    return M

def learn(M,I,O):
    M.fit(I,O)


def testModel(M,X_test,y_test):
    pred = M.predict_proba(X_test)[:,1]
    res = pd.DataFrame({"Id": list(range(len(pred))), "2014": pred, "2015": pred})
    print(roc_auc_score(y_test.iloc[:,0],res.iloc[:,0]))
    print(roc_auc_score(y_test.iloc[:,1],res.iloc[:,1]))
    
def writeOutput(M,name):
    prediction = M.predict_proba(Goal)[:,1]
    resultat = pd.DataFrame({"Id": GoalTemp["Id"], "2014": prediction, "2015": prediction})
    resultat.to_csv("result/" + (name + ".csv"), sep=";", columns=["Id", "2014", "2015"], index=False)

#faire tourner plein de fois le même modèle et sélectionner le meilleur.
def naturalSelection(n):
    print("TODO man")

#Featuring(In)
#Featuring(Goal)


#X_train, X_test, y_train, y_test = CutData(In, Out,0)
#M = Logit()
#M.fit(X_train, y_train)
#writeOutput(M,"ceciestuntestdebrute")

def launch(N,ratio):
    for i in range(N):
        X_train, X_test, y_train, y_test = CutData(In, Out,ratio)
        M = Logit()
        M.fit(X_train, y_train)
        testModel(M,X_test,y_test)
        testModel(M,In,Out)
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

