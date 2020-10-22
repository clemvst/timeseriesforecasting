import torch
from torchsummary import summary
from torch.utils.data import Dataset,DataLoader
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

from dataset import load_prepare, boxplot, normalize, split_window, DatasetSales

from Model import ConvNet, train_fn, test_fn


# Time Series Prediction

# Clemence Vast - 2020
# Objectives

# We want to predict the next month for items and shops

#     load and clean data
#     visualize different data representations
#     select interesting data
#     build a CNN
#     train one or multiple items-shop series in the CNN
#     evaluate the CNN and analyze results

# general values
path_csv= "./sales_train.csv"
window = 12 # number of months in the sliding window

# training values
learning_rate = 0.001
epochs = 200

# evaluation values
prediction = []
batch_size = 2
iterations =  int(inputs.shape[0]/2)
    
    
def main():
    ## Load DataFrame and visualize info
    train_csv = load_prepare(path_csv)
    print(train_csv.head(10))
    print(train_csv.info())
    print(train_csv.shape)
    
    # Preparing and cleaning data
    ## let's remove negative values for sales
    train_csv.query('item_price > 0', inplace=True)
    print(train_csv.shape)
    
    # We want to check if there are some rare exceptions in the values and filter them
    # Box plot
    boxplot(train_csv, "Boxplot of sales")
    
    # We see that values tend to be under 1000 except 2 values. 
    # To clean a little bit our dataset we're going to remove those values.
    train_csv.query('item_cnt_day < 1000', inplace=True)
    
    # group items by date blocks
    df_date = pd.DataFrame(train_csv.groupby(["date_block_num", "shop_id", "item_id", "item_price"], as_index=False)['item_cnt_day'].sum())
    print(df_date.head(5))

    # Visualize by Month
    
    plt.plot( "date_block_num", "item_cnt_day", data = df_date.sort_values(by="date_block_num"))

    plt.xticks(rotation=90)

    plt.show()
    
    #Normalizing
    
    train_csv["item_csv_day_norm"] = normalize(train_csv["item_cnt_day"])
    
    print(train_csv)
    
    # Cleaning
    # Let's check the data and filter when we see unuseful values
    train_csv = train_csv.groupby(['date_block_num',"shop_id", "item_id"]).agg({"item_cnt_day_norm":"count"}, as_index=False)
    print(train_csv)
    
    ## We could group all values together, by date block

    train_grouped = train_csv.sort_values('date_block_num').groupby(['date_block_num'], as_index=True)
    train_grouped = train_grouped.agg({'item_cnt_day_norm':"mean"})
    train_grouped.rename(columns={"item_cnt_day_norm":"sales_norm"}, inplace=True)
    print(train_grouped.head())
    
    
    ## Split values between train and test
    test_proportion = 0.4
    train, test = train_test_split(train_grouped, test_size=test_proportion)

    print('Train proportion : {:.0f}%'.format((1-test_proportion)*100))
    print('Validation proportion : {:.0f}%'.format(test_proportion*100))

    train_x,train_y = split_window(train.sales_norm.values,window)
    valid_x,valid_y = split_window(test.sales_norm.values,window)
    
    ## preparing Train
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    cnn = ConvNet().to(device)
    optimizer = torch.optim.Adam(cnn.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()
    
    # Torch summary gives a small summary of the Neural Network with the input size

    print(summary(cnn,[12,1]))
    
    ## Create sets and Data Loaders

    trainset = DatasetSales(train_x.reshape(train_x.shape[0],train_x.shape[1],1),train_y)
    validset = DatasetSales(valid_x.reshape(valid_x.shape[0],valid_x.shape[1],1),valid_y)
    trainloader = torch.utils.data.DataLoader(trainset,batch_size=2,shuffle=False)
    validationloader = torch.utils.data.DataLoader(trainset,batch_size=2,shuffle=False)

    dataset_loaders = {"train": trainloader, "test": validationloader}

    
    # Train model
    train_losses = []
    valid_losses = []
    
    for epoch in range(epochs):
        print('\nEpochs {}/{}'.format(epoch+1,epochs))
        train_fn()
        test_fn()
    # Evaluation of the model

    # We want to print the Loss for both training and validation, to check if the 
    # model learned correctly from the features and detect overfitting.
    
    plt.plot(train_losses,label='Loss train')
    plt.plot(valid_losses,label='Loss validation')
    plt.title('MSE Loss graph')
    plt.legend(bbox_to_anchor=(1.05, 1), loc= 'center left', borderaxespad=0.)
    plt.show()

    targetX , targetY = split_window(train.sales_norm.values,window)
    inputs = targetX.reshape(targetX.shape[0],targetX.shape[1],1)

    cnn.eval()

    for i in range(iterations):
        preds = cnn(torch.tensor(inputs[batch_size*i:batch_size*(i+1)]).float())
        prediction.append(preds.detach().numpy())
        
    fig, ax = plt.subplots(1, 2,figsize=(11,4))
    ax[0].set_title('Prediction')
    ax[0].plot(prediction)
    ax[1].set_title('Reality')
    ax[1].plot(targetY)
    plt.show()
    
    
# Analysis

# Of course my model is not really representing the reality and is not complete, I would normally need to create a model by item-shop duo and then connect them all together for example. To work with a bigger dataset and add additional values I could also use a k-fold technique. Maybe convolutional layers are also not the best way to predict this kind of information. Overall, the Loss looks good but I'm not sure it would with a different dataset of values.

# If I had more time here's what I wouldve done:

#     Build a model for each item-shop duo
#     Connect them all with a fully connected layer
#     Build a model by shop (for all items)
#     Build a model by item (for all shops)

# I could've compare all those models and select my favorite one or analyse them too.
