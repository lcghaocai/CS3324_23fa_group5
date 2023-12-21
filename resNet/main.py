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
batch_size = 8
num_workers = 8
split_rate = 0.9
lr = [1e-4, 1e-4]
freeze_epochs = 5
freezing = 0
epochs = 10
fold_size = 10
k_size = 10
weight = ResNet50_Weights.IMAGENET1K_V2


def change_freeze_status(freeze):
    for param in net.resnet.parameters():
        param.requires_grad = not freeze

    for param in net.resnet.fc.parameters():
        param.requires_grad = True

def train(epoch, freeze=False):
    net.train()
    train_loss = 0
    true_label = []
    pred_label = []
    if freeze:
        optimizer = freeze_optimizer
    else:
        optimizer = unfreeze_optimizer
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

def eval(epoch, freeze = False):
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
    if freeze:
        freeze_scheduler.step()
    else:
        unfreeze_scheduler.step()
    acc = np.sum(true_label == pred_label) / len(pred_label)
    print('Epoch: {}, Validation Loss:{:6f}, Accuracy: {:6f}'.format(epoch, val_loss, acc))
    return acc

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
    size_after_qfe = len(raw_data)
    raw_size = size_after_qfe // qfe_size
    train_size = int(raw_size * split_rate)
    test_size = raw_size - train_size
    raw_index = randperm(raw_size).tolist()
    raw_index = sum(list(map(lambda x:[qfe_size*x, qfe_size*x+1, qfe_size*x+2], raw_index)), [])
    used_size = len(raw_index)
    slice_size = used_size // fold_size
    

    
    
    for cur_fold in range(k_size):
        if cur_fold == 0:
            train_data = torch.utils.data.Subset(raw_data, raw_index[slice_size:])
            test_data = torch.utils.data.Subset(raw_data, raw_index[:slice_size])
        else:
            train_data = torch.utils.data.Subset(raw_data, raw_index[:slice_size * cur_fold] + raw_index[slice_size * (cur_fold + 1):])
            test_data = torch.utils.data.Subset(raw_data, raw_index[slice_size * cur_fold:slice_size * (cur_fold + 1)])
        # train_data, test_data = random_split(raw_data, [train_size, test_size])
        train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=num_workers, drop_last=True)
        test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False, num_workers=num_workers)
        net = ResNet50(weight)
        net.cuda()
        freeze_optimizer = optim.Adam(net.parameters(), lr = lr[0])
        unfreeze_optimizer = optim.Adam(net.parameters(), lr = lr[1])
        freeze_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer=freeze_optimizer, T_0=10, T_mult=2)
        unfreeze_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer=unfreeze_optimizer, T_0=10, T_mult=2)
        criterion = nn.CrossEntropyLoss()
        max_acc = 0
        change_freeze_status(True)

        for epoch in range(freeze_epochs):
            train(epoch + 1, True)
            acc = eval(epoch + 1, True)
            if acc > max_acc:
                save_path = f'./resnet50_imagenet_v2_{cur_fold}.pth'
                max_acc = acc
                log.info('max_acc = {:6f}, fold = {}'.format(acc, cur_fold))
                torch.save(net, save_path)
        
        change_freeze_status(False)
        for epoch in range(epochs):
            train(epoch + 1)
            acc = eval(epoch + 1)
            if acc > max_acc:
                save_path = f'./resnet50_imagenet_v2_{cur_fold}.pth'
                max_acc = acc
                log.info('max_acc = {:6f}, fold = {}'.format(acc, cur_fold))
                torch.save(net, save_path)
        
        save_path = f'./resnet50_imagenet_v2_{datetime.now().timestamp()}_{cur_fold}.pth'
        torch.save(net, save_path)
    
    
