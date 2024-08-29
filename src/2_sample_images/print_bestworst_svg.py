
import sys
import numpy as np
from datetime import datetime 
import pandas as pd
import re

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torch.backends.cudnn as cudnn
import torchvision
from torchvision import datasets, transforms

import matplotlib.pyplot as plt
from imagenetv2_pytorch import ImageNetV2Dataset

import os
import argparse
from PIL import Image

# download and create datasets
valid_dataset = datasets.MNIST(root='mnist_data', 
                               train=False)


mnist_best1 = valid_dataset.data[8413]
mnist_best2 = valid_dataset.data[6657]
mnist_worst1 = valid_dataset.data[7846]
mnist_worst2 = valid_dataset.data[249]

title1 = "High #1"
title2 = "High #2"
title3 = "Low #1"
title4 = "Low #2"

mnist_fig = plt.figure(dpi=300)
plt.subplot(1,4,1)
plt.axis('off')
plt.imshow(mnist_best1, cmap='gray_r')
plt.title(title1,fontsize=7)
plt.subplot(1,4,2)
plt.axis('off')
plt.imshow(mnist_best2, cmap='gray_r')
plt.title(title2,fontsize=7)
plt.subplot(1,4,3)
plt.axis('off')
plt.imshow(mnist_worst1, cmap='gray_r')
plt.title(title3,fontsize=7)
plt.subplot(1,4,4)
plt.axis('off')
plt.imshow(mnist_worst2, cmap='gray_r')
plt.title(title4,fontsize=7)

plt.savefig('mnist_bestworst.svg',bbox_inches='tight')

# download and create datasets
valid_dataset = datasets.KMNIST(root='kmnist_data', 
                               train=False)



kmnist_best1 = valid_dataset.data[5184]
kmnist_best2 = valid_dataset.data[4151]
kmnist_worst1 = valid_dataset.data[5174]
kmnist_worst2 = valid_dataset.data[9683]

title1 = "High #1"
title2 = "High #2"
title3 = "Low #1"
title4 = "Low #2"

kmnist_fig = plt.figure(dpi=300)
plt.subplot(1,4,1)
plt.axis('off')
plt.imshow(kmnist_best1, cmap='gray_r')
plt.title(title1,fontsize=7)
plt.subplot(1,4,2)
plt.axis('off')
plt.imshow(kmnist_best2, cmap='gray_r')
plt.title(title2,fontsize=7)
plt.subplot(1,4,3)
plt.axis('off')
plt.imshow(kmnist_worst1, cmap='gray_r')
plt.title(title3,fontsize=7)
plt.subplot(1,4,4)
plt.axis('off')
plt.imshow(kmnist_worst2, cmap='gray_r')
plt.title(title4,fontsize=7)

plt.savefig('kmnist_bestworst.svg',bbox_inches='tight')


# download and create datasets
valid_dataset = datasets.FashionMNIST(root='fashionmnist_data', 
                               train=False)


fashionmnist_best1 = valid_dataset.data[4341]
fashionmnist_best2 = valid_dataset.data[2157]
fashionmnist_worst1 = valid_dataset.data[6774]
fashionmnist_worst2 = valid_dataset.data[4144]

title1 = "High #1"
title2 = "High #2"
title3 = "Low #1"
title4 = "Low #2"

fashionmnist_fig = plt.figure(dpi=300)
plt.subplot(1,4,1)
plt.axis('off')
plt.imshow(fashionmnist_best1, cmap='gray_r')
plt.title(title1,fontsize=7)
plt.subplot(1,4,2)
plt.axis('off')
plt.imshow(fashionmnist_best2, cmap='gray_r')
plt.title(title2,fontsize=7)
plt.subplot(1,4,3)
plt.axis('off')
plt.imshow(fashionmnist_worst1, cmap='gray_r')
plt.title(title3,fontsize=7)
plt.subplot(1,4,4)
plt.axis('off')
plt.imshow(fashionmnist_worst2, cmap='gray_r')
plt.title(title4,fontsize=7)

plt.savefig('fashionmnist_bestworst.svg',bbox_inches='tight')


valid_dataset = datasets.SVHN(root='svhn_data', 
                               split="test")


svhn_best1 = valid_dataset[21183][0]
svhn_best2 = valid_dataset[18596][0]
svhn_worst1 = valid_dataset[24946][0]
svhn_worst2 = valid_dataset[9990][0]

svhn_fig = plt.figure(dpi=300)
plt.subplot(1,4,1)
plt.axis('off')
plt.imshow(svhn_best1)
plt.title(title1,fontsize=7)
plt.subplot(1,4,2)
plt.axis('off')
plt.imshow(svhn_best2)
plt.title(title2,fontsize=7)
plt.subplot(1,4,3)
plt.axis('off')
plt.imshow(svhn_worst1)
plt.title(title3,fontsize=7)
plt.subplot(1,4,4)
plt.axis('off')
plt.imshow(svhn_worst2)
plt.title(title4,fontsize=7)

plt.savefig('svhn_bestworst.svg',bbox_inches='tight')


valid_dataset = datasets.CIFAR10(root='cifar_data', 
                               train=False)


cifar_best1 = valid_dataset.data[4014]
cifar_best2 = valid_dataset.data[3625]
cifar_worst1 = valid_dataset.data[6427]
cifar_worst2 = valid_dataset.data[4443]

cifar_fig = plt.figure(dpi=300)
plt.subplot(1,4,1)
plt.axis('off')
plt.imshow(cifar_best1)
plt.title(title1,fontsize=7)
plt.subplot(1,4,2)
plt.axis('off')
plt.imshow(cifar_best2)
plt.title(title2,fontsize=7)
plt.subplot(1,4,3)
plt.axis('off')
plt.imshow(cifar_worst1)
plt.title(title3,fontsize=7)
plt.subplot(1,4,4)
plt.axis('off')
plt.imshow(cifar_worst2)
plt.title(title4,fontsize=7)

plt.savefig('cifar_bestworst.svg',bbox_inches='tight')


valid_dataset = datasets.ImageNet(root='inet',split='val',transform=None)

imagenet_best1 = valid_dataset[19612][0]
imagenet_best2 = valid_dataset[11009][0]
imagenet_worst1 = valid_dataset[30765][0]
imagenet_worst2 = valid_dataset[47853][0]

# 700 x 469
# 800 x 1200
# 530 x 194
# 380 x 500

imagenet_fig, (i0, i1, i2, i3) = plt.subplots(1, 4, gridspec_kw={'width_ratios': [700/469, 800/1200, 530/194, 380/500]})
i0.imshow(imagenet_best1)
i0.axis('off')
i0.set_title(title1, fontsize=7)
i1.imshow(imagenet_best2)
i1.axis('off')
i1.set_title(title2, fontsize=7)
i2.imshow(imagenet_worst1)
i2.axis('off')
i2.set_title(title3, fontsize=7)
i3.imshow(imagenet_worst2)
i3.axis('off')
i3.set_title(title4, fontsize=7)
plt.savefig('imagenet_bestworst.svg',bbox_inches='tight',dpi=300)

valid_dataset = ImageNetV2Dataset("matched-frequency",transform=None)

imagenetv2_best1 = valid_dataset[5139][0]
imagenetv2_best2 = valid_dataset[9889][0]
imagenetv2_worst1 = valid_dataset[6944][0]
imagenetv2_worst2 = valid_dataset[8465][0]

imagenet_fig = plt.figure()
plt.subplot(1,4,1)
plt.axis('off')
plt.imshow(imagenetv2_best1)
plt.title(title1,fontsize=7)
plt.subplot(1,4,2)
plt.axis('off')
plt.imshow(imagenetv2_best2)
plt.title(title2,fontsize=7)
plt.subplot(1,4,3)
plt.axis('off')
plt.imshow(imagenetv2_worst1)
plt.title(title3,fontsize=7)
plt.subplot(1,4,4)
plt.axis('off')
plt.imshow(imagenetv2_worst2)
plt.title(title4,fontsize=7)

plt.savefig('imagenetv2_bestworst.svg',bbox_inches='tight',dpi=300)

