import sys
import numpy as np
from datetime import datetime 
import pandas as pd
import re

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import models, datasets, transforms
import matplotlib.pyplot as plt
# from imagenetv2_pytorch import ImageNetV2Dataset

start_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

N_TRAIN = 1281167
N_VALID = 50000
N_TEST = 10000
BATCH_SIZE = 1000
DEVICE = 'cpu'

model_names = ["alexnet","densenet201","googlenet","convnext_tiny","convnext_large",
	"efficientnet_b0","efficientnet_b3","efficientnet_b7","regnet_y_128gf",
	"inception_v3","squeezenet1_1","mobilenet_v3_small","mobilenet_v3_large",
	"resnet50","resnet101","vgg19_bn","resnext101_32x8d","resnext101_64x4d",
	"wide_resnet101_2","densenet121"]

m = model_names[int(sys.argv[1])]

transform = transforms.Compose([            #[1]
 transforms.Resize(256),                    #[2]
 transforms.CenterCrop(224),                #[3]
 transforms.ToTensor(),                     #[4]
 transforms.Normalize(                      #[5]
 mean=[0.485, 0.456, 0.406],                #[6]
 std=[0.229, 0.224, 0.225]                  #[7]
 )])

def get_actuals(model,n):
	actuals = []
	model.eval()
	for i in range(n):
		actuals.append(dataset[i][1])
	return(actuals)

def get_probs(model, data_loader, device):
    '''
    Function for computing the accuracy of the predictions over the entire data_loader
    '''
    probabilities = []
    with torch.no_grad():
        model.eval()
        for X, y_true in data_loader:
            X = X.to(device)
            y_prob = model(X).softmax(dim=1)
            probabilities.append(y_prob)
            print (datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    return probabilities

train_dataset = datasets.ImageNet(root='inet',split='train',transform=transform)
#valid_dataset = datasets.ImageNet(root='inet',split='val',transform=transform)
#test_dataset = ImageNetV2Dataset("matched-frequency",transform=transform)

train_loader = DataLoader(dataset=train_dataset, 
                          batch_size=BATCH_SIZE, 
                          shuffle=False)

my_model_fun = getattr(models,m)
my_model = my_model_fun(pretrained=True)

results_train = get_probs(my_model, train_loader, DEVICE)

prob_array = results_train[0].detach().numpy()

for i in range(1,len(results_train)):
	prob_array = np.concatenate((prob_array,results_train[i].detach().numpy()),axis=0)

# results = np.append(results_valid,results_test)
np.savetxt(m + "_batch_probs.csv",prob_array,delimiter="\t")

end_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

file = open(m + "_start_end_times.csv", "a")
file.write('Start time: ' + start_time + '\n')
file.write('End time: ' + end_time + '\n')
file.close()
