import pandas as pd

class Reading:

    def __init__(self):

        self.train = pd.read_csv("data/inputTrain.csv")
        self.output = pd.read_csv("data/outputTrain.csv")
        self.test = pd.read_csv("data/inputTest.csv")

        
