import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics

cols_labels = ["AccXMean", "AccXSD", "AccXSkew", "AccXKurtosis", "AccXMin", "AccXMax", "AccYMean", "AccYSD", "AccYSkew", "AccYKurtosis", "AccYMin", "AccYMax", "AccZMean", "AccZSD", "AccZSkew", "AccZKurtosis", "AccZMin", "AccZMax",
               "GyrXMean", "GyrXSD", "GyrXSkew", "GyrXKurtosis", "GyrXMin", "GyrXMax", "GyrYMean", "GyrYSD", "GyrYSkew", "GyrYKurtosis", "GyrYMin", "GyrYMax", "GyrZMean", "GyrZSD", "GyrZSkew", "GyrZKurtosis", "GyrZMin", "GyrZMax"]

data = pd.read_csv("../data/training_dataset_normalise")
data.drop(457, axis=0, inplace=True)
data.drop(columns=["Unnamed: 0"], axis=1, inplace=True)

X = data.loc[:, data.columns != "TypeOfShot"]
Y = data["TypeOfShot"]

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.33)

clf = RandomForestClassifier(n_estimators=100, max_depth=10)

clf.fit(X_train, Y_train)

y_pred = clf.predict(X_test)

print("Accuracy : ", metrics.accuracy_score(Y_test, y_pred))

new_stroke = pd.read_csv("../data/test_prediction_normalize")
new_stroke.drop(columns=["Unnamed: 0"], axis=1, inplace=True)

print(new_stroke)

typeofshot = clf.predict(new_stroke)
print(typeofshot)