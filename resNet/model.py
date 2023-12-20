import os
import cv2
import torch
import random
import numpy as np
from PIL import Image
import pandas as pd
from torch import nn
from torch.utils.data import Dataset, DataLoader, random_split
import torchvision.models as models
from torchvision import transforms
from torchvision.models import resnet50, ResNet50_Weights
from torchvision.models import densenet121,DenseNet121_Weights

filedir = '.'
weight = ResNet50_Weights.IMAGENET1K_V2
weight1= DenseNet121_Weights.IMAGENET1K_V1
qfe_size = 3

net_list=['resnet50','densenet121','othernet']

def qfe(image, file_name, transf):
    res_list = []
    for i in range(qfe_size):
        res_list.append([transf(image), file_name])
    return res_list


class model:
    def __init__(self):
        self.resnet_checkpoint = "resnet50_imagenet_v2.pth"
        self.densenet_checkpoint = "densenet121_imagenet.pth"
        self.device = torch.device("cpu")
        self.resnet_model = None
        self.densenet_model = None

    def load(self, dir_path):
        """
        load the model and weights.
        dir_path is a string for internal use only - do not remove it.
        all other paths should only contain the file name, these paths must be
        concatenated with dir_path, for example: os.path.join(dir_path, filename).
        make sure these files are in the same directory as the model.py file.
        :param dir_path: path to the submission directory (for internal use only).
        :return:
        """
        # # join paths
        # checkpoint_path = os.path.join(dir_path, self.checkpoint)
        # self.model = torch.load(checkpoint_path, map_location=self.device)
        # self.model.to(self.device)
        # self.model.eval()

         # Load ResNet50 model
        resnet_checkpoint_path = os.path.join(dir_path, self.resnet_checkpoint)
        self.resnet_model = torch.load(resnet_checkpoint_path, map_location=self.device)
        self.resnet_model.to(self.device)
        self.resnet_model.eval()

         # Load DenseNet121 model
        densenet_checkpoint_path = os.path.join(dir_path, self.densenet_checkpoint)
        self.densenet_model = torch.load(densenet_checkpoint_path, map_location=self.device)
        self.densenet_model.to(self.device)
        self.densenet_model.eval()






    def predict(self, input_image):
        """
        perform the prediction given an image.
        input_image is a ndarray read using cv2.imread(path_to_image, 1).
        note that the order of the three channels of the input_image read by cv2 is BGR.
        :param input_image: the input image to the model.
        :return: an int value indicating the class for the input image
        """
      
        # Transform the input image
        transf = transforms.Compose([transforms.ToPILImage(), transforms.ToTensor(), transforms.RandomRotation(180), weight.transforms()])
        image = torch.unsqueeze(transf(input_image), 0)

        # Perform prediction using both models
        with torch.no_grad():
            resnet_score = self.resnet_model(image)
            densenet_score = self.densenet_model(image)

        # Voting mechanism
        resnet_pred = torch.argmax(resnet_score, 1).item()
        densenet_pred = torch.argmax(densenet_score, 1).item()

        # If both models agree, return the prediction
        if resnet_pred == densenet_pred:
            return resnet_pred
        else:
            # If models disagree, i don't know what to do
            return resnet_pred




class ResNet50(nn.Module):
    def __init__(self, weight):
        super(ResNet50, self).__init__()
        self.resnet = models.resnet50(weight)
        num_features = self.resnet.fc.in_features
        self.resnet.fc = nn.Linear(in_features=num_features, out_features=2)

    def forward(self, x):
        x = self.resnet(x)
        return x
    

class DenseNet121(nn.Module):
    def __init__(self, weight1):
        super(DenseNet121, self).__init__()
        self.densenet = models.densenet121(weight1)
        in_features = self.densenet.classifier.in_features
        #drop out level
        self.densenet.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(in_features, 2)
        )

    def forward(self, x):
        return self.densenet(x)



class MySet(Dataset):
    def __init__(self, data_path, label_file, enable_transform = True):
        self.current = 0
        data_path = os.path.normpath(data_path)
        label_file = os.path.normpath(label_file)
        self.data = []
        if enable_transform:
            transf = transforms.Compose([ transforms.ToPILImage(), transforms.RandomRotation(180), transforms.ToTensor(), weight.transforms()])
            for file_name in os.listdir(data_path):
                self.data += qfe(cv2.imread(os.path.join(data_path, file_name)), file_name, transf)
        else:
            for file_name in os.listdir(data_path):
                self.data.append([cv2.imread(os.path.join(data_path, file_name)), file_name])
        
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

       
if __name__ == '__main__':
    m = model()
    m.load(".")

  

    correct_cnt = 0
    tot = 0
    loader = MySet( filedir + "/1-Images/1-Training Set", filedir + "/2-Groundtruths/HRDC Hypertensive Classification Training Labels.csv", enable_transform=False)




    for i in range(loader.size):
        image, label = loader[i]
        result = m.predict(image)
        tot += 1
        if result == label:
            print('success')
            correct_cnt += 1
        else:
            print('failure')
    print(f'{correct_cnt}, {tot}, correct_rate:{correct_cnt/tot}')