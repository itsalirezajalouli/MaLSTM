#   Imports
import torch
import numpy as np
import pandas as pd 
import torch.nn as nn
from copy import deepcopy
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import Dataset 

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

hist = 7
newData = lookBack(data, hist, 'Close')
npData = newData.to_numpy()

#   Scale
scaler = MinMaxScaler(feature_range = (-1, 1))
npData = scaler.fit_transform(npData)

#   Feats & Targets
X = npData[:, 1:]
X = deepcopy(np.flip(X, axis = 1))     #   Because LSTM goes from oldest to latest summary
y = npData[:, 0]                       #   Which is start from -7 to -1

#   Split
splitIdx = int(len(X) * 0.95)
xTrain = X[:splitIdx]
xTest = X[splitIdx:]
yTrain = y[:splitIdx]
yTest = y[splitIdx:]

#   PyTorch LSTM requires extra dimention so here we go...
xTrain = xTrain.reshape((-1, hist, 1))
xTest = xTest.reshape((-1, hist, 1))
yTrain = yTrain.reshape((-1, 1))
yTest = yTest.reshape((-1, 1))

#   Move 2 torch
xTrain = torch.tensor(xTrain).float()
xTest = torch.tensor(xTest).float()
yTrain = torch.tensor(yTrain).float()
yTest = torch.tensor(yTest).float()

#   Dataset
class timeSeriesDataset(Dataset):
    def __init__(self, X, y) -> None:
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, i):
        return self.X[i], self.y[i]

trainSet = timeSeriesDataset(xTrain, yTrain)
testSet = timeSeriesDataset(xTest, yTest)
