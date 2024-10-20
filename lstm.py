#   Imports
import torch
import numpy as np
import pandas as pd 
import torch.nn as nn
from copy import deepcopy
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import Dataset, DataLoader

#   Device
device = 'cuda' if torch.cuda.is_available() else 'cpu'

#   Hyperz
batchSize = 16
learningRate = 1e-3
numEpochs = 10

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

#   Dataset & loader
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
trainLoader = DataLoader(trainSet, batchSize, True)
testLoader = DataLoader(testSet, batchSize, True)

#   Batching!
for _, batch in enumerate(trainLoader):
    xBatch, yBatch = batch[0].to(device), batch[1].to(device)

#   Model
class LSTM(nn.Module):
    def __init__(self, inputSize: int, hiddenSize: int, nStackedLayers: int) -> None:
        super().__init__()
        self.hiddenSize = hiddenSize
        self.nStackedLayers = nStackedLayers 
        self.lstm = nn.LSTM(inputSize, hiddenSize, nStackedLayers, batch_first = True)
        self.fc = nn.Linear(hiddenSize, 1)

    def forward(self, X) -> None:
        batchSize = X.size(0)
        h0 = torch.zeros(self.nStackedLayers, batchSize, self.hiddenSize).to(device)
        c0 = torch.zeros(self.nStackedLayers, batchSize, self.hiddenSize).to(device)
        out, _ = self.lstm(X, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out

model = LSTM(1, 4, 1)
model.to(device)

#   Optim & loss
lossFn = nn.MSELoss()
optim = torch.optim.Adam(model.parameters(), lr = learningRate)

#   Train loop
def trainOneEpoch():
    model.train()
    runningLoss = 0.0
    print(f'Epoch: {epoch + 1}')
    for batchIdx, batch in enumerate(trainLoader):
        xBatch, yBatch = batch[0].to(device), batch[1].to(device)

        output = model(xBatch)
        loss = lossFn(output, yBatch)
        runningLoss += loss.item()

        optim.zero_grad()
        loss.backward()
        optim.step()                            #   take a slight step toward the gradient

        if batchIdx % 100 == 99:
            avgBatchLoss = runningLoss / 100
            print(f'Batch {batchIdx + 1}, Batch Loss: {avgBatchLoss:.3f}')
            runningLoss = 0.0

#   Test loop
def validOneEpoch():
    model.eval()
    runningLoss = 0.0
    for _, batch in enumerate(testLoader):
        xBatch, yBatch = batch[0].to(device), batch[1].to(device)

        output = model(xBatch)
        loss = lossFn(output, yBatch)
        runningLoss += loss.item()

    avgBatchLoss = runningLoss / len(testLoader)
    print(f'Val Loss: {avgBatchLoss:.3f}')

    print('\u2589' * 80, '\n')

#   Main loop
for epoch in range(numEpochs):
    trainOneEpoch()
    validOneEpoch()

#   Reverse convert
trainPreds = predicted.flatten()
dummies = np.zeros((xTrain.shape[0], hist + 1))
dummies[:, 0] = trainPreds 
dummies = scaler.inverse_transform(dummies)
trainPreds = deepcopy(dummies[:, 0])
print(trainPreds)

#   Plotting
with torch.no_grad():
    predicted = model(xTrain.to(device)).to('cpu').numpy()
plt.plot(yTrain, label = 'Actual Close')
plt.plot(predicted, label = 'Predicted Close')
plt.xlabel('Day')
plt.ylabel('Close')
plt.legend()
plt.show()
