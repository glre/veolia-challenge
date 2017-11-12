import pandas as pd
from sklearn.model_selection import train_test_split
import itertools as it

# Cut data into training set and test set, to do cross validation
# Return a training set, a test set
# ratio is the proportion we put in the test set. 0 = no test set
def CutData(X,Y,ratio):    
    indexOnes = [i for i in range(len(Y)) if (Y.iloc[i]["2014"] == 1 or Y.iloc[i]["2015"] == 1)]
    indexNotOnes = list(set(range(len(Y))) - set(indexOnes))

    XOnes = X.iloc[indexOnes]
    YOnes = Y.iloc[indexOnes]
    XNotOnes = X.iloc[indexNotOnes]
    YNotOnes = Y.iloc[indexNotOnes]
    
    XOnes_train, XOnes_test, YOnes_train, YOnes_test = train_test_split(XOnes, YOnes, test_size=ratio, random_state=None)
    XNotOnes_train, XNotOnes_test, YNotOnes_train, YNotOnes_test = train_test_split(XNotOnes, YNotOnes, test_size=ratio, random_state=None)
    
    X_train = pd.concat([XOnes_train,XNotOnes_train])
    X_test = pd.concat([XOnes_test,XNotOnes_test])

    Y_train = pd.concat([YOnes_train,YNotOnes_train])
    Y_test = pd.concat([YOnes_test,YNotOnes_test])

    return (X_train, X_test, Y_train, Y_test)
