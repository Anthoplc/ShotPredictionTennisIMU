from scipy.signal import find_peaks
from scipy.stats import kurtosis, skew
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
from sklearn import preprocessing

cols_labels = ["AccXMean", "AccXSD", "AccXSkew", "AccXKurtosis", "AccXMin", "AccXMax", "AccYMean", "AccYSD", "AccYSkew", "AccYKurtosis", "AccYMin", "AccYMax", "AccZMean", "AccZSD", "AccZSkew", "AccZKurtosis", "AccZMin", "AccZMax",
               "GyrXMean", "GyrXSD", "GyrXSkew", "GyrXKurtosis", "GyrXMin", "GyrXMax", "GyrYMean", "GyrYSD", "GyrYSkew", "GyrYKurtosis", "GyrYMin", "GyrYMax", "GyrZMean", "GyrZSD", "GyrZSkew", "GyrZKurtosis", "GyrZMin", "GyrZMax"]

frequence = 60
pourcentage_prominence=50 #pour l'homme 40 c'est bien
intervalle=30

new_shot_data = pd.read_csv("../data/Xsens DOT0_20230318_101827_209.csv", skiprows=range(0,11), usecols=range(2,8))
new_shot_data.columns = ["AccX", "AccY", "AccZ", "GyrX", "GyrY", "GyrZ"]
new_shot_data=new_shot_data.iloc[:-2,:]
print(new_shot_data)


main="gauche"
if main=="gaucher":
    new_shot_data['AccY'] = new_shot_data['AccY'].apply(lambda x: -x if x > 0 else abs(x))
    new_shot_data['AccY'] = new_shot_data['AccY'].apply(lambda x: x if x > 0 else -abs(x))
    new_shot_data['GyrY'] = new_shot_data['GyrY'].apply(lambda x: -x if x > 0 else abs(x))
    new_shot_data['GyrY'] = new_shot_data['GyrY'].apply(lambda x: x if x > 0 else -abs(x))

maxX=np.max(new_shot_data["AccX"])
seuil_prominence=maxX*pourcentage_prominence/100
AccX_peaks = find_peaks(new_shot_data["AccX"],prominence=seuil_prominence,distance=40)
#plt.plot(new_shot_data["AccX"])
#plt.show()
peaks = AccX_peaks[0]


NB_shots = len(AccX_peaks[0])

to_predict_shot = pd.DataFrame(columns=cols_labels)

for i in peaks:
    data_new_shot = new_shot_data[i-intervalle:i+intervalle]
    row = list()
    for j in range(6):
        mean = np.mean(data_new_shot.iloc[:, j])
        sd = np.std(data_new_shot.iloc[:, j])
        skewness = skew(data_new_shot.iloc[:, j])
        kurtosisness = kurtosis(data_new_shot.iloc[:, j])
        minimum = np.min(data_new_shot.iloc[:, j])
        maximum = np.max(data_new_shot.iloc[:, j])
        row.append(mean)
        row.append(sd)
        row.append(skewness)
        row.append(kurtosisness)
        row.append(minimum)
        row.append(maximum)
    to_predict_shot.loc[len(to_predict_shot)] = row

print(to_predict_shot)


data = pd.read_csv("../data/training_dataset")
data.drop(457, axis=0, inplace=True)
data.drop(columns=["Unnamed: 0"], axis=1, inplace=True)

X = data.loc[:, data.columns != "TypeOfShot"]
Y = data["TypeOfShot"]

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.33)

clf = RandomForestClassifier(n_estimators=200,criterion="gini",max_depth=10,min_samples_split=5,min_samples_leaf=1,max_features=10)

clf.fit(X_train, Y_train)

y_pred = clf.predict(X_test)

print("Accuracy : ", metrics.accuracy_score(Y_test, y_pred))

typeofshot = clf.predict(to_predict_shot)
print(typeofshot)

to_predict_shot = to_predict_shot.assign(TypeOfShot=pd.Series(typeofshot))

import os

# Obtenir le nom de fichier de new_shot_data
filename = os.path.basename("../data/Xsens DOT0_20230318_101827_209.csv")
# Supprimer l'extension .csv du nom de fichier
filename_parts = filename.split("_")
filename = filename_parts[2]
filename = os.path.splitext(filename)[0]
# Ajouter l'extension .csv au nom de fichier
to_predict_filename = f"{filename}_to_predict.csv"

# Enregistrer to_predict_shot en tant que fichier CSV avec le nom de fichier correspondant
to_predict_shot.to_csv(f"../data_xsens/{to_predict_filename}", index=False)



