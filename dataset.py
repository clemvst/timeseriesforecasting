import torch
from torchsummary import summary
from torch.utils.data import Dataset,DataLoader
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

from sklearn.preprocessing import MinMaxScaler

### Different dataset functions


def load_prepare(path_csv):
    train_csv = pd.read_csv(path_csv, parse_dates=['date'])
    return train_csv

def boxplot(df, name):
    fig1, ax1 = plt.subplots()
    ax1.set_title(name)
    ax1.boxplot(df.item_cnt_day, vert=False)
    plt.show()
    
def normalize(df_col):
    data = np.array(df_col)
    scaler = MinMaxScaler(feature_range=(0, 1))
    normalized = scaler.fit_transform(data.reshape(-1, 1))
    return normalized


def split_window(window, n_steps):
    # split to build a window
    X, y = list(), list()
    for i in range(len(window)):
        # find the end of this pattern
        end_ix = i + n_steps
        # check if we are beyond the window
        if end_ix > len(window)-1:
            break
        # gather input and output parts of the pattern
        seq_x, seq_y = window[i:end_ix], window[end_ix]
        X.append(seq_x)
        y.append(seq_y)
    return np.array(X), np.array(y)



class DatasetSales(Dataset):
    ## Creates the Dataset and extracts it
    def __init__(self,item_group,target):
        self.item_group = item_group
        self.target = target
    
    def __len__(self):
        return len(self.item_group)
    
    def __getitem__(self,idx):
        item = self.item_group[idx]
        label = self.target[idx]
        return item,label