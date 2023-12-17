import os
from PIL import Image
import cv2
import torch
import random
import logging
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch import randperm
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
from torchvision.models import resnet50, ResNet50_Weights
from torchvision.models import densenet121,DenseNet121_Weights

filedir = '.'
save_path = './densenet121_imagenet.pth'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
batch_size = 16
num_workers =4 
split_rate = 0.8
lr = 1e-4
epochs = 10
weight = DenseNet121_Weights.IMAGENET1K_V1
first = True
qfe_size = 4



# 训练集图像路径
train_image_dir = "/kaggle/input/trainingset/1-Images/1-Training Set/"

# 图像尺寸
image_size = (224, 224)

# 用于计算均值和标准差的变量
mean = np.zeros(3)
std = np.zeros(3)
num_images = 0


# 遍历训练集图像
for image_file in os.listdir(train_image_dir):
    if image_file.endswith(".png"):
        # 加载图像并转换为 NumPy 数组
        image_path = os.path.join(train_image_dir, image_file)
        image = Image.open(image_path).convert("RGB")
        image = image.resize(image_size)
        image_array = np.array(image) / 255.0  # 将像素值缩放到 [0, 1] 范围内

        # 更新均值和标准差
        mean += np.mean(image_array, axis=(0, 1))
        std += np.std(image_array, axis=(0, 1))
        num_images += 1

# 计算均值和标准差
mean /= num_images
std /= num_images

print("均值：", mean)
print("标准差：", std)


def preprocess(image):
    # Convert the image to PIL Image format
    image = Image.fromarray(image)

    # Resize the image to the required input size of DenseNet
    image = image.resize(image_size)

    # Convert the image to tensor
    image = transforms.ToTensor()(image)

    # Normalize the image
    normalize = transforms.Normalize(mean, std)
    image = normalize(image)

    return image



def qfe(image, file_name, transf):
    res_list = []
    for i in range(qfe_size):
        res_list.append([transf(image), file_name])
    return res_list
    
class MySet(Dataset):
    def __init__(self, data_path, label_file):
        self.current = 0
        data_path = os.path.normpath(data_path)
        label_file = os.path.normpath(label_file)
        self.data = []
        transf = transforms.Compose([preprocess, transforms.RandomRotation(180), weight.transforms(antialias=True)])
        #transf = transforms.Compose([ transforms.ToPILImage(), transforms.RandomRotation(180), transforms.ToTensor(), weight.transforms(antialias=True)])
        for file_name in os.listdir(data_path):
            self.data += qfe(cv2.imread(os.path.join(data_path, file_name)), file_name, transf)
        self.labels = pd.read_csv(label_file)
        print(self.labels.shape)
        self.labels.set_index('Image', inplace=True)
        self.size = len(self.data)

    def __getitem__(self, index):
        image = self.data[index]
        return image[0], self.labels.loc[image[1], 'Hypertensive']

    def retrieve(self, index):
        return self.data[index][1]
        
    def __len__(self):
        return self.size

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
 
if __name__ == '__main__':
    log = logging.getLogger()
    log.setLevel(logging.INFO) # Log等级总开关
    logfile = './log.txt'
    fh = logging.FileHandler(logfile, mode='a') # open的打开模式这里可以进行参考 
    formatter = logging.Formatter("%(asctime)s - line:%(lineno)d - %(levelname)s==> %(message)s")
    fh.setFormatter(formatter)
    log.addHandler(fh)  
    
    raw_data = MySet( "/kaggle/input/trainingset/1-Images/1-Training Set",  "/kaggle/input/groundtruths/2-Groundtruths/HRDC Hypertensive Classification Training Labels.csv")
    train_size = int(len(raw_data) * split_rate)
    test_size = len(raw_data) - train_size
    raw_index = randperm(len(raw_data)).tolist()
    train_data = torch.utils.data.Subset(raw_data, raw_index[:train_size])
    test_data = torch.utils.data.Subset(raw_data, raw_index[train_size:])
    # train_data, test_data = random_split(raw_data, [train_size, test_size])
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=num_workers, drop_last=True)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    
    
    
    
    net = densenet121(pretrained=True)
    # 获取原始DenseNet的输入通道数
    in_features = net.classifier.in_features

    
    # 创建新的全连接层，输出类别数为2（假设有两个类别）
    net.classifier = nn.Sequential(
    nn.Dropout(0.5),  # 添加Dropout层，丢弃概率为0.5
    nn.Linear(in_features, 2)
    )
    net.cuda()
    optimizer = optim.Adam(net.parameters(), lr = lr)
    decayRate = 0.96
    my_lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=optimizer, gamma=decayRate)
    criterion = nn.CrossEntropyLoss()
    
   
       
    for epoch in range(epochs):
        train(epoch + 1)
        eval(epoch + 1)
    torch.save(net, save_path)

    