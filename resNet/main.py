import os
from PIL import Image
import cv2
import torch
import random
from datetime import datetime
import logging
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from model import *
from torch import randperm
from torchvision import transforms, models
from torchvision.models import resnet50, ResNet50_Weights

filedir = '.'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
batch_size = 16
num_workers = 8
split_rate = 0.9
lr = 1e-4
epochs = 15
fold_size = 10
weight = ResNet50_Weights.IMAGENET1K_V2
first = True


def train(epoch):
    net.train()
    train_loss = 0
    true_label = []
    pred_label = []
    for data, label in train_loader:
        data, label = data.cuda(), label.cuda()
        optimizer.zero_grad()
        output = net(data)
        pred = torch.argmax(output, 1)
        true_label.append(label.cpu().data.numpy())
        pred_label.append(pred.cpu().data.numpy())
        loss = criterion(output, label)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()*data.size(0)
    train_loss /= len(train_loader.dataset)
    true_label, pred_label = np.concatenate(true_label), np.concatenate(pred_label)
    acc = np.sum(true_label == pred_label) / len(pred_label)
    print('Epoch: {}, Train Loss: {:6f}, accuracy: {:6f}'.format(epoch, train_loss, acc))    

def eval(epoch):
    net.eval()
    val_loss = 0
    true_label = []
    pred_label = []
    with torch.no_grad():
        for data, label in test_loader:
            data, label = data.cuda(), label.cuda()
            output = net(data)
            pred = torch.argmax(output, 1)
            true_label.append(label.cpu().data.numpy())
            pred_label.append(pred.cpu().data.numpy())
            loss = criterion(output, label)
            val_loss += loss.item()*data.size(0)
    val_loss /= len(test_loader.dataset)
    true_label, pred_label = np.concatenate(true_label), np.concatenate(pred_label)
    for i in range(len(true_label)):
        eq = true_label[i] == pred_label[i]
        file_name = raw_data.retrieve(train_size + i)
        log.info(f'Epoch: {epoch}, file name: {file_name}, success: {eq}')
    my_lr_scheduler.step()
    acc = np.sum(true_label == pred_label) / len(pred_label)
    print('Epoch: {}, Validation Loss:{:6f}, Accuracy: {:6f}'.format(epoch, val_loss, acc))

def log_init():
    log = logging.getLogger()
    log.setLevel(logging.INFO) # Log等级总开关
    logfile = './log_resnet.txt'
    fh = logging.FileHandler(logfile, mode='w') # open的打开模式这里可以进行参考 
    formatter = logging.Formatter("%(asctime)s - line:%(lineno)d - %(levelname)s==> %(message)s")
    fh.setFormatter(formatter)
    log.addHandler(fh)
    return log

if __name__ == '__main__':
    log = log_init()
    
    raw_data = MySet( filedir + "/1-Images/1-Training Set", filedir + "/2-Groundtruths/HRDC Hypertensive Classification Training Labels.csv")
    raw_size = len(raw_data) // qfe_size
    train_size = int(raw_size * split_rate)
    test_size = raw_size - train_size
    raw_index = randperm(raw_size).tolist()
    raw_index = sum(list(map(lambda x:[qfe_size*x, qfe_size*x+1, qfe_size*x+2], raw_index)), [])
    slice_data = []
    slice_size = raw_size // fold_size
    for i in range(fold_size):
        slice_data.append(torch.utils.data.Subset(raw_data, raw_index[slice_size * i:slice_size * (i + 1)]))
    train_data = torch.utils.data.Subset(raw_data, raw_index[:])
    test_data = torch.utils.data.Subset(raw_data, raw_index[train_size:])
    # train_data, test_data = random_split(raw_data, [train_size, test_size])
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=num_workers, drop_last=True)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    net = ResNet50(weight)
    net.cuda()
    optimizer = optim.Adam(net.parameters(), lr = lr)
    decayRate = 0.96
    my_lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=optimizer, gamma=decayRate)
    criterion = nn.CrossEntropyLoss()
    
   
       
    for epoch in range(epochs):
        train(epoch + 1)
        eval(epoch + 1)
    
    save_path = f'./resnet50_imagenet_v2_{datetime.now().timestamp()}.pth'
    torch.save(net, save_path)
    
    
