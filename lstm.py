#   Imports
import torch
import numpy as np
import pandas as pd 
import torch.nn as nn
import matplotlib.pyplot as plt
from copy import deepcopy

#   Device
device = 'cuda' if torch.cuda.is_available() else 'cpu'

#   Data
data = pd.read_csv('./AMZN.csv')
data = data[['Date', 'Close']]
data['Date'] = pd.to_datetime(data['Date'])

#   Plot
# plt.plot(data['Date'], data['Close'])
# plt.show()

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
print(newData)

