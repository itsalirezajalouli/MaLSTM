#   Imports
import torch
import numpy as np
import pandas as pd 
import torch.nn as nn
from copy import deepcopy
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

#   Device
device = 'cuda' if torch.cuda.is_available() else 'cpu'

#   Data
data = pd.read_csv('./AMZN.csv')
data = data[['Date', 'Close']]
data['Date'] = pd.to_datetime(data['Date'])

#   Lookback function
def lookBack(df, nSteps: int, label: str) -> pd.DataFrame:
    df = deepcopy(df)
    df.set_index('Date', inplace = True)
    for i in range(1, nSteps + 1):
        df[f'{label}(t-{i})'] = df[f'{label}'].shift(i)
    #   Drop Sun & Sat
    df.dropna(inplace = True)
    return df

newData = lookBack(data, 7, 'Close')
npData = newData.to_numpy()

#   Scale
scaler = MinMaxScaler(feature_range = (-1, 1))
npData = scaler.fit_transform(npData)

#   Feats & Targets
featz = npData[:, 1:]
featz = deepcopy(np.flip(featz, axis = 1)) #   Because LSTM goes from oldest to latest history summary
targz = npData[:, 0]                       #   Which is start from -7 to -1

#   Split
splitIdx = int(len(featz) * 0.95)
xTrain = featz[:splitIdx]
xTest = featz[splitIdx:]
yTrain = targz[:splitIdx]
yTest = targz[splitIdx:]

#   PyTorch LSTM requires extra dimention
