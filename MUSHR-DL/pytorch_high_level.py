import os
import cv2
import numpy as np 
from tqdm import tqdm
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

GPU = False
if torch.cuda.is_available():
    # torch.cuda.device_count -> for counting number of gpus. most laptops will have just 1
    device = torch.device("cuda:0")# currently only supporting 1 gpu
    GPU = True
    print('running on GPU')
else:
    device = torch.device("cpu")
    GPU = False
    print("running on CPU")

def fwd_pass(net,X,Y,optimizer,loss_function,train=False):
    if train:
        optimizer.zero_grad()
    output = net(X)
    if(output.shape != Y.shape):
        print("output shape does not match target shape!")
        print("input shape:",X.shape)
        print("output shape:",output.shape)
        print("target shape:",Y.shape)
        exit()
    loss = loss_function(output,Y)
    if train:
        loss.backward()
        optimizer.step()
    return loss

def fit(net,
        X,
        Y,
        train_log,
        validation_set,
        BATCH_SIZE,
        EPOCHS,
        model_name = None):

    print(model_name)
    assert model_name != None, "model name cannot be None"

    optimizer = optim.Adam(net.parameters(), lr=0.001)
    loss_function = nn.MSELoss()

    val_size = int(validation_set*len(X))
    data_size = len(X)
    train_size = data_size - val_size

    elasped_time = 0

    for epochs in range(EPOCHS):
        start_time = time.time()

        #insample data
        train_average_loss = 0
        val_average_loss = 0
        train_counter = 0
        val_counter = 0
        for i in tqdm(range(0,train_size, BATCH_SIZE ) ):
            batch_X = (X[i:i+BATCH_SIZE]).to(device)
            batch_Y = (Y[i:i+BATCH_SIZE]).to(device)
            train_loss = fwd_pass(net,batch_X,batch_Y,optimizer,loss_function,train=True)
            if i%100==0:
                train_average_loss += float(train_loss.cpu())
                train_counter += 1

        with torch.no_grad():
            for i in tqdm(range(train_size,data_size,BATCH_SIZE)):
                batch_X = (X[i:i+BATCH_SIZE]).to(device)
                batch_Y = (Y[i:i+BATCH_SIZE]).to(device)
                val_loss = fwd_pass(net,batch_X,batch_Y,optimizer,loss_function,train=False)
                if i%10==0:
                    val_average_loss += float(val_loss.cpu())
                    val_counter += 1


        end_time = time.time()
        elasped_time += end_time - start_time
        print('epoch:{} \t;elasped time: {}'.format(epochs,elasped_time))
        print('train loss:\t', train_average_loss / train_counter)
        print('val loss:\t', val_average_loss / val_counter)
        train_log.append([train_average_loss / train_counter, val_average_loss / val_counter, elasped_time])  # [train_loss,val_loss,elasped_time]

        print(net.state_dict()['fc4.bias'])
        state = {'net': net.state_dict(), 'optimizer': optimizer.state_dict(), 'epoch': epochs}
        torch.save(state, "{}_batch_{}_epoch_{}.pth".format(model_name, BATCH_SIZE, epochs))
        torch.cuda.empty_cache()

    return train_log       




