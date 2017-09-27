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

def prediction(M, X, Y, name):
    prediction = M.predict_proba(Goal)[:,1]
    resultat = pd.DataFrame({"Id": GoalTemp["Id"], "2014": prediction, "2015": prediction})
    resultat.to_csv("result/" + (name + ".csv"), sep=";", columns=["Id", "2014", "2015"], index=False)

