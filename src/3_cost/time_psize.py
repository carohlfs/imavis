import sys
import os
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

from ptflops import get_model_complexity_info
from fvcore.nn import FlopCountAnalysis

macs_vec = []
params_vec = []
flops_vec = []

inputs = (torch.randn((1,3,224,224)),)

net = models.resnext101_32x8d()
net.eval()
macs, params = get_model_complexity_info(net, (3, 224, 224), as_strings=True,print_per_layer_stat=True, verbose=True)
macs_vec.append(macs)
params_vec.append(params)
flops = FlopCountAnalysis(net,inputs)
flops_vec.append(flops.total())

net = models.wide_resnet101_2()
net.eval()
macs, params = get_model_complexity_info(net, (3, 224, 224), as_strings=True,print_per_layer_stat=True, verbose=True)
macs_vec.append(macs)
params_vec.append(params)
flops = FlopCountAnalysis(net,inputs)
flops_vec.append(flops.total())

net = models.efficientnet_b0()
net.eval()
macs, params = get_model_complexity_info(net, (3, 224, 224), as_strings=True,print_per_layer_stat=True, verbose=True)
macs_vec.append(macs)
params_vec.append(params)
flops = FlopCountAnalysis(net,inputs)
flops_vec.append(flops.total())

net = models.efficientnet_b7()
net.eval()
macs, params = get_model_complexity_info(net, (3, 224, 224), as_strings=True,print_per_layer_stat=True, verbose=True)
macs_vec.append(macs)
params_vec.append(params)
flops = FlopCountAnalysis(net,inputs)
flops_vec.append(flops.total())

net = models.resnet101()
net.eval()
macs, params = get_model_complexity_info(net, (3, 224, 224), as_strings=True,print_per_layer_stat=True, verbose=True)
macs_vec.append(macs)
params_vec.append(params)
flops = FlopCountAnalysis(net,inputs)
flops_vec.append(flops.total())

net = models.densenet201()
net.eval()
macs, params = get_model_complexity_info(net, (3, 224, 224), as_strings=True,print_per_layer_stat=True, verbose=True)
macs_vec.append(macs)
params_vec.append(params)
flops = FlopCountAnalysis(net,inputs)
flops_vec.append(flops.total())

net = models.vgg19_bn()
net.eval()
macs, params = get_model_complexity_info(net, (3, 224, 224), as_strings=True,print_per_layer_stat=True, verbose=True)
macs_vec.append(macs)
params_vec.append(params)
flops = FlopCountAnalysis(net,inputs)
flops_vec.append(flops.total())

net = models.mobilenet_v3_large()
net.eval()
macs, params = get_model_complexity_info(net, (3, 224, 224), as_strings=True,print_per_layer_stat=True, verbose=True)
macs_vec.append(macs)
params_vec.append(params)
flops = FlopCountAnalysis(net,inputs)
flops_vec.append(flops.total())

net = models.mobilenet_v3_small()
net.eval()
macs, params = get_model_complexity_info(net, (3, 224, 224), as_strings=True,print_per_layer_stat=True, verbose=True)
macs_vec.append(macs)
params_vec.append(params)
flops = FlopCountAnalysis(net,inputs)
flops_vec.append(flops.total())

net = models.googlenet()
net.eval()
macs, params = get_model_complexity_info(net, (3, 224, 224), as_strings=True,print_per_layer_stat=True, verbose=True)
macs_vec.append(macs)
params_vec.append(params)
flops = FlopCountAnalysis(net,inputs)
flops_vec.append(flops.total())

net = models.inception_v3()
net.eval()
macs, params = get_model_complexity_info(net, (3, 224, 224), as_strings=True,print_per_layer_stat=True, verbose=True)
macs_vec.append(macs)
params_vec.append(params)
flops = FlopCountAnalysis(net,inputs)
flops_vec.append(flops.total())

net = models.alexnet()
net.eval()
macs, params = get_model_complexity_info(net, (3, 224, 224), as_strings=True,print_per_layer_stat=True, verbose=True)
macs_vec.append(macs)
params_vec.append(params)
flops = FlopCountAnalysis(net,inputs)
flops_vec.append(flops.total())

>>> params_vec
['88.79 M', '126.89 M', '5.29 M', '66.35 M', '44.55 M', '20.01 M', '143.68 M', '5.48 M', '2.54 M', '13.0 M', '27.16 M', '61.1 M']
>>> 

>>> macs_vec
['16.51 GMac', '22.82 GMac', '401.67 MMac', '5.27 GMac', '7.85 GMac', '4.37 GMac', '19.7 GMac', '228.36 MMac', '60.13 MMac', '1.51 GMac', '2.85 GMac', '715.56 MMac']

>>> flops_vec
[16476534784, 22795599872, 401659392, 5266983936, 7833969664, 4340870784, 19661791744, 226033824, 59649536, 1504878688, 2847268992, 714197696]



class LeNet5(nn.Module):
    def __init__(self, n_classes):
        super(LeNet5, self).__init__()       
        self.feature_extractor = nn.Sequential(            
            nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5, stride=1),
            nn.Tanh(),
            nn.AvgPool2d(kernel_size=2),
            nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5, stride=1),
            nn.Tanh(),
            nn.AvgPool2d(kernel_size=2),
            nn.Conv2d(in_channels=16, out_channels=120, kernel_size=5, stride=1),
            nn.Tanh()
        )
        self.classifier = nn.Sequential(
            nn.Linear(in_features=120, out_features=84),
            nn.Tanh(),
            nn.Linear(in_features=84, out_features=n_classes),
        )
    def forward(self, x):
        x = self.feature_extractor(x)
        x = torch.flatten(x, 1)
        logits = self.classifier(x)
        probs = F.softmax(logits, dim=1)
        return logits, probs



model = LeNet5(10)
model.eval()
inputs = (torch.randn((1,1,32,32)),)
macs, params = get_model_complexity_info(model, (1, 32, 32), as_strings=True,print_per_layer_stat=True, verbose=True)
macs_vec.append(macs)
params_vec.append(params)
flops = FlopCountAnalysis(model,inputs)
flops_vec.append(flops.total())

## go into different folder
os.chdir('../Effort_Jul2022/pytorch-cifar-master')

from models import *

model = SimpleDLA()
model.eval()
inputs = (torch.randn((1,3,32,32)),)
macs, params = get_model_complexity_info(model, (3, 32, 32), as_strings=True,print_per_layer_stat=True, verbose=True)
macs_vec.append(macs)
params_vec.append(params)
flops = FlopCountAnalysis(model,inputs)
flops_vec.append(flops.total())

model = ResNet18()
model.eval()
macs, params = get_model_complexity_info(model, (3, 32, 32), as_strings=True,print_per_layer_stat=True, verbose=True)
macs_vec.append(macs)
params_vec.append(params)
flops = FlopCountAnalysis(model,inputs)
flops_vec.append(flops.total())

import pandas as pd

model_names = ['resnext101_32x8d','wide_resnet101_2','efficientnet_b0','efficientnet_b7',
	'resnet101','densenet201','vgg19_bn','mobilenet_v3_large','mobilenet_v3_small',
	'googlenet','inception_v3','alexnet','LeNet5','SimpleDLA','ResNet18']

data = {'Model': model_names,
		'macs': macs_vec,
		'params': params_vec,
		'flops': flops_vec}

os.chdir('../../Pretrained/')

df = pd.DataFrame(data)
df.to_csv('flops.csv')