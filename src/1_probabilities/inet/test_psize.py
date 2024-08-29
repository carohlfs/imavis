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

model_names = ["alexnet","densenet201","googlenet","convnext_tiny","convnext_large",
	"efficientnet_b0","efficientnet_b3","efficientnet_b7","regnet_y_128gf",
	"inception_v3","squeezenet1_1","mobilenet_v3_small","mobilenet_v3_large",
	"resnet50","resnet101","vgg19_bn","resnext101_32x8d","resnext101_64x4d",
	"wide_resnet101_2","densenet121"]

m = model_names[int(sys.argv[1])]
PSIZE = int(sys.argv[2])

transform = transforms.Compose([            #[1]
 transforms.Resize(PSIZE),                  #[2]
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

def get_probs(model, dataset, n, run_name):
	probabilities = []
	model.eval()
	for i in range(n):
		casei = dataset[i][0]
		y_true = dataset[i][1]
		batchi = torch.unsqueeze(casei, 0)
		probi = model(batchi).softmax(dim=1)
		prob, label = torch.max(probi, 1)
		probabilities.append([prob.item(),label.item()])
		if i % 1000 == 0:
			print (datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
			print(str(run_name) + " iteration " + str(i))
	return probabilities

valid_dataset = datasets.ImageNet(root='inet',split='val',transform=transform)
test_dataset = ImageNetV2Dataset("matched-frequency",transform=transform)

my_model_fun = getattr(models,m)
my_model = my_model_fun(pretrained=True)

results_valid = get_probs(my_model, valid_dataset, N_VALID, m + str(PSIZE) + " valid")
results_test = get_probs(my_model, test_dataset, N_TEST, m + str(PSIZE) + " test")

results = np.append(results_valid,results_test)
np.savetxt(m + "." + str(PSIZE) + ".csv",results,delimiter="\t")


