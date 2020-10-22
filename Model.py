import torch
from torchsummary import summary
from torch.utils.data import Dataset,DataLoader
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

## NEED TO DEFINE DATALOADERS AND CNN BEFORE USING THIS FILE


class ConvNet(nn.Module):
    ## CNN with two Conv 1 D
    def __init__(self):
        super(ConvNet,self).__init__()
        self.conv1 = nn.Conv1d(12,100,kernel_size=1)
        self.conv2 = nn.Conv1d(100,64,kernel_size=1)
        self.relu = nn.ReLU(inplace=True)
        self.fc1 = nn.Linear(64*2,50)
        self.fc2 = nn.Linear(50,1)
        
    def forward(self,x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = x.view(-1)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        
        return x
    

def train_fn():
    """ Train part with dataloader and train loss"""
    train = dataset_loaders["train"]
    running_loss = .0
    
    cnn.train()
    
    for idx, (inputs,labels) in enumerate(train):
        inputs = inputs.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        preds = cnn(inputs.float())
        loss = criterion(preds,labels.float())
        loss.backward()
        optimizer.step()
        running_loss += loss
        
    train_loss = running_loss/len(train)
    train_losses.append(train_loss.detach().numpy())
    
    print(f'train_loss : {train_loss}')
    
def test_fn():
    """ Test part with dataloader and test loss"""
    test = dataset_loaders["test"]
    running_loss = .0
    cnn.eval()
    
    with torch.no_grad():
        for idx, (inputs, labels) in enumerate(test):
            inputs = inputs.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            preds = cnn(inputs.float())
            loss = criterion(preds,labels)
            running_loss += loss
            
        valid_loss = running_loss/len(test)
        valid_losses.append(valid_loss.detach().numpy())
        print(f'valid_loss : {valid_loss}')
        