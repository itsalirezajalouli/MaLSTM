#   Imports
import torch
import numpy as np
import pandas as pd 
import torch.nn as nn
import matplotlib.pyplot as plt

#   Device
device = 'cuda' if torch.cuda.is_available() else 'cpu'

#   Data
data = pd.read_csv('./AMZN.csv')
data = data[['Date', 'Close']]
data['Date'] = pd.to_datetime(data['Date'])

#   Plot
plt.plot(data['Date'], data['Close'])
plt.show()
