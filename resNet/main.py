import os
from PIL import Image
import torch
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
import torchvision.models as models

filedir = '.'
save_path = './resnet50.pth'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
batch_size = 8
num_workers = 8
split_rate = 0.9
lr = 1e-3
epochs = 20

class MySet(Dataset):
    def __init__(self, data_path, label_file):
        self.current = 0
        data_path = os.path.normpath(data_path)
        label_file = os.path.normpath(label_file)
        self.data = []
        transf = transforms.ToTensor()
        for file_name in os.listdir(data_path):
            self.data.append([transf(Image.open(os.path.join(data_path, file_name)).convert('RGB')), file_name])
        self.labels = pd.read_csv(label_file)
        print(self.labels.shape)
        self.labels.set_index('Image', inplace=True)
        self.size = len(self.data)

    def __getitem__(self, index):
        image = self.data[index]
        return image[0], self.labels.loc[image[1], 'Hypertensive']
    def __len__(self):
        return self.size

def train(epoch):
    net.train()
    train_loss = 0
    for data, label in train_loader:
        data, label = data.cuda(), label.cuda()
        optimizer.zero_grad()
        output = net(data)
        loss = criterion(output, label)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()*data.size(0)
    train_loss /= len(train_loader.dataset)
    print('Epoch: {}, Train Loss:{:6f}'.format(epoch, train_loss))    

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
    acc = np.sum(true_label == pred_label) / len(pred_label)
    print('Epoch: {}, Validation Loss:{:6f}, Accuracy: {:6f}'.format(epoch, val_loss, acc))
 
if __name__ == '__main__':
    raw_data = MySet( filedir + "/1-Images/1-Training Set", filedir + "/2-Groundtruths/HRDC Hypertensive Classification Training Labels.csv")
    train_size = int(len(raw_data) * split_rate)
    test_size = len(raw_data) - train_size
    train_data, test_data = random_split(raw_data, [train_size, test_size])
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=False, num_workers=num_workers, drop_last=True)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    net = models.resnet50()
    net.cuda()
    optimizer = optim.Adam(net.parameters(), lr = lr)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        train(epoch + 1)
        eval(epoch + 1)
    torch.save(net, save_path)
    
    
