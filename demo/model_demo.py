import os
import cv2
import torch
import random
import pandas as pd
from torch import nn
import torchvision.models as models

filedir = '..'

class model:
    def __init__(self):
        self.checkpoint = "model_weights.pth"
        self.device = torch.device("cpu")

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
        self.model = ResNet34(num_classes=2)
        # join paths
        checkpoint_path = os.path.join(dir_path, self.checkpoint)
        self.model.load_state_dict(torch.load(checkpoint_path, map_location=self.device))
        self.model.to(self.device)
        self.model.eval()

    def predict(self, input_image):
        """
        perform the prediction given an image.
        input_image is a ndarray read using cv2.imread(path_to_image, 1).
        note that the order of the three channels of the input_image read by cv2 is BGR.
        :param input_image: the input image to the model.
        :return: an int value indicating the class for the input image
        """
        image = cv2.resize(input_image, (512, 512))
        image = image / 255
        image = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0)
        image = image.to(self.device, torch.float)

        with torch.no_grad():
            score = self.model(image)
        _, pred_class = torch.max(score, 1)

        pred_class = pred_class.detach().cpu()
        pred_class = int(pred_class)
        return pred_class


class ResNet34(nn.Module):
    def __init__(self, num_classes=2, pretrained=False):
        super(ResNet34, self).__init__()
        self.resnet = models.resnet34(pretrained=pretrained)
        num_features = self.resnet.fc.in_features
        self.resnet.fc = nn.Linear(in_features=num_features, out_features=num_classes)

    def forward(self, x):
        x = self.resnet(x)
        return x

class dataSet:
    def __init__(self, data_path, label_file):
        self.current = 0
        data_path = os.path.normpath(data_path)
        label_file = os.path.normpath(label_file)
        self.data = []
        for file_name in os.listdir(data_path):
            self.data.append([cv2.imread(os.path.join(data_path, file_name)), file_name])
        random.shuffle(self.data)
        self.labels = pd.read_csv(label_file)
        print(self.labels.shape)
        self.labels.set_index('Image', inplace=True)
        self.size = len(self.data)

    def __getitem__(self, index):
        image = self.data[index]
        return image[0], self.labels.loc[image[1], 'Hypertensive']

    
        
if __name__ == '__main__':
    m = model()
    m.load(".")
    correct_cnt = 0
    tot = 0
    loader = dataSet( filedir + "/1-Images/1-Training Set", filedir + "/2-Groundtruths/HRDC Hypertensive Classification Training Labels.csv")
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
