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
from imagenetv2_pytorch import ImageNetV2Dataset

N_TRAIN = 1281167
N_VALID = 50000
N_TEST = 10000

transform = transforms.Compose([            #[1]
 transforms.Resize(256),                    #[2]
 transforms.CenterCrop(224),                #[3]
 transforms.ToTensor(),                     #[4]
 transforms.Normalize(                      #[5]
 mean=[0.485, 0.456, 0.406],                #[6]
 std=[0.229, 0.224, 0.225]                  #[7]
 )])

train_dataset = datasets.ImageNet(root='inet',split='train',transform=transform)
train_actuals = []
for i in range(N_TRAIN):
	train_actuals.append(train_dataset[i][1])
	if i % 1000 == 0:
		print (datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
		print("train iteration " + str(i))

valid_dataset = datasets.ImageNet(root='inet',split='val',transform=transform)
valid_actuals = []
for i in range(N_VALID):
	valid_actuals.append(valid_dataset[i][1])
	if i % 1000 == 0:
		print (datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
		print("valid iteration " + str(i))

test_dataset = ImageNetV2Dataset("matched-frequency",transform=transform)
test_actuals = []
for i in range(N_TEST):
	test_actuals.append(test_dataset[i][1])
	if i % 1000 == 0:
		print (datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
		print("test iteration " + str(i))

np.savetxt("train_actuals.csv",train_actuals,delimiter="\t")
np.savetxt("valid_actuals.csv",valid_actuals,delimiter="\t")
np.savetxt("test_actuals.csv",test_actuals,delimiter="\t")


