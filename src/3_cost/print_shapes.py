import sys, os
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

valid_dataset = datasets.ImageNet(root='../data/imagenet',split='val',transform=None)
valid_shapes = []
for i in range(N_VALID):
	valid_shapes.append(valid_dataset[i][0].size)
	if i % 1000 == 0:
		print (datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
		print("valid iteration " + str(i))

# ImageNetV2 Dataset
original_wd = os.getcwd()
os.chdir('../data/imagenetv2')
test_dataset = ImageNetV2Dataset("matched-frequency",transform=None)
test_shapes = []
for i in range(N_TEST):
	test_shapes.append(test_dataset[i][0].size)
	if i % 1000 == 0:
		print (datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
		print("test iteration " + str(i))

os.chdir(original_wd)

np.savetxt("valid_shapes.txt",valid_shapes,delimiter="\t")
np.savetxt("test_shapes.txt",test_shapes,delimiter="\t")


