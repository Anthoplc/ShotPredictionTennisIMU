from scipy.signal import find_peaks
from scipy.stats import kurtosis, skew
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

cols_labels = ["AccXMean", "AccXSD", "AccXSkew", "AccXKurtosis", "AccXMin", "AccXMax", "AccYMean", "AccYSD", "AccYSkew", "AccYKurtosis", "AccYMin", "AccYMax", "AccZMean", "AccZSD", "AccZSkew", "AccZKurtosis", "AccZMin", "AccZMax",
               "GyrXMean", "GyrXSD", "GyrXSkew", "GyrXKurtosis", "GyrXMin", "GyrXMax", "GyrYMean", "GyrYSD", "GyrYSkew", "GyrYKurtosis", "GyrYMin", "GyrYMax", "GyrZMean", "GyrZSD", "GyrZSkew", "GyrZKurtosis", "GyrZMin", "GyrZMax"]

frequence = 30
new_shot_data = pd.read_csv("../data/G_0_0_0_0.csv", skiprows=range(0,11), usecols=range(2,8))
new_shot_data.columns = ["AccX", "AccY", "AccZ", "GyrX", "GyrY", "GyrZ"]

AccX_peaks = find_peaks(new_shot_data["AccX"], prominence=60, distance=30)
peaks = AccX_peaks[0]

NB_shots = len(AccX_peaks[0])

to_predict_shot = pd.DataFrame(columns=cols_labels)

for i in peaks:
    data_new_shot = new_shot_data[i-frequence:i+frequence]
    row = list()
    for j in range(6):
        data_new_shot.iloc[:, j] = (data_new_shot.iloc[:, j]-np.min(data_new_shot.iloc[:, j]))/(np.max(data_new_shot.iloc[:, j])-np.min(data_new_shot.iloc[:, j]))
        print(data_new_shot.iloc[:, j])
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