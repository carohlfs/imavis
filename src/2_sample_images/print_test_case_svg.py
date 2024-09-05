
import numpy as np
from datetime import datetime 
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from torchvision import datasets, transforms

import matplotlib.pyplot as plt
from Pillow import Image

# download and create datasets
valid_dataset = datasets.MNIST(root='mnist_data', 
                               train=False)


mnist_fig1 = transforms.Resize((32,32))(transforms.Resize((28,28))(valid_dataset.data))[0]
mnist_fig2 = transforms.Resize((32,32))(transforms.Resize((14,14))(valid_dataset.data))[0]
mnist_fig4 = transforms.Resize((32,32))(transforms.Resize((7,7))(valid_dataset.data))[0]
mnist_fig7 = transforms.Resize((32,32))(transforms.Resize((4,4))(valid_dataset.data))[0]
mnist_fig14 = transforms.Resize((32,32))(transforms.Resize((2,2))(valid_dataset.data))[0]

title1 = "28x28"
title2 = "14x14"
title3 = "7x7"
title4 = "4x4"
title5 = "2x2"

mnist_fig = plt.figure(dpi=300)
plt.subplot(1,5,1)
plt.axis('off')
plt.imshow(mnist_fig1, cmap='gray_r')
plt.title(title1,fontsize=7)
plt.subplot(1,5,2)
plt.axis('off')
plt.imshow(mnist_fig2, cmap='gray_r')
plt.title(title2,fontsize=7)
plt.subplot(1,5,3)
plt.axis('off')
plt.imshow(mnist_fig4, cmap='gray_r')
plt.title(title3,fontsize=7)
plt.subplot(1,5,4)
plt.axis('off')
plt.imshow(mnist_fig7, cmap='gray_r')
plt.title(title4,fontsize=7)
plt.subplot(1,5,5)
plt.axis('off')
plt.imshow(mnist_fig14, cmap='gray_r')
plt.title(title5,fontsize=7)

plt.savefig('mnist_fig.svg',bbox_inches='tight')

# download and create datasets
valid_dataset = datasets.KMNIST(root='kmnist_data', 
                               train=False)


kmnist_fig1 = transforms.Resize((32,32))(transforms.Resize((28,28))(valid_dataset.data))[0]
kmnist_fig2 = transforms.Resize((32,32))(transforms.Resize((14,14))(valid_dataset.data))[0]
kmnist_fig4 = transforms.Resize((32,32))(transforms.Resize((7,7))(valid_dataset.data))[0]
kmnist_fig7 = transforms.Resize((32,32))(transforms.Resize((4,4))(valid_dataset.data))[0]
kmnist_fig14 = transforms.Resize((32,32))(transforms.Resize((2,2))(valid_dataset.data))[0]

kmnist_fig = plt.figure(dpi=300)
plt.subplot(1,5,1)
plt.axis('off')
plt.imshow(kmnist_fig1, cmap='gray_r')
plt.title(title1,fontsize=7)
plt.subplot(1,5,2)
plt.axis('off')
plt.imshow(kmnist_fig2, cmap='gray_r')
plt.title(title2,fontsize=7)
plt.subplot(1,5,3)
plt.axis('off')
plt.imshow(kmnist_fig4, cmap='gray_r')
plt.title(title3,fontsize=7)
plt.subplot(1,5,4)
plt.axis('off')
plt.imshow(kmnist_fig7, cmap='gray_r')
plt.title(title4,fontsize=7)
plt.subplot(1,5,5)
plt.axis('off')
plt.imshow(kmnist_fig14, cmap='gray_r')
plt.title(title5,fontsize=7)

plt.savefig('kmnist_fig.svg',bbox_inches='tight')


# download and create datasets
valid_dataset = datasets.FashionMNIST(root='fashionmnist_data', 
                               train=False)


fashionmnist_fig1 = transforms.Resize((32,32))(transforms.Resize((28,28))(valid_dataset.data))[0]
fashionmnist_fig2 = transforms.Resize((32,32))(transforms.Resize((14,14))(valid_dataset.data))[0]
fashionmnist_fig4 = transforms.Resize((32,32))(transforms.Resize((7,7))(valid_dataset.data))[0]
fashionmnist_fig7 = transforms.Resize((32,32))(transforms.Resize((4,4))(valid_dataset.data))[0]
fashionmnist_fig14 = transforms.Resize((32,32))(transforms.Resize((2,2))(valid_dataset.data))[0]

fashionmnist_fig = plt.figure(dpi=300)
plt.subplot(1,5,1)
plt.axis('off')
plt.imshow(fashionmnist_fig1, cmap='gray_r')
plt.title(title1,fontsize=7)
plt.subplot(1,5,2)
plt.axis('off')
plt.imshow(fashionmnist_fig2, cmap='gray_r')
plt.title(title2,fontsize=7)
plt.subplot(1,5,3)
plt.axis('off')
plt.imshow(fashionmnist_fig4, cmap='gray_r')
plt.title(title3,fontsize=7)
plt.subplot(1,5,4)
plt.axis('off')
plt.imshow(fashionmnist_fig7, cmap='gray_r')
plt.title(title4,fontsize=7)
plt.subplot(1,5,5)
plt.axis('off')
plt.imshow(fashionmnist_fig14, cmap='gray_r')
plt.title(title5,fontsize=7)

plt.savefig('fashionmnist_fig.svg',bbox_inches='tight')


valid_dataset = datasets.SVHN(root='svhn_data', 
                               split="test")


title1 = "32x32"
title2 = "16x16"
title3 = "8x8"
title4 = "4x4"
title5 = "2x2"

svhn_fig1 = np.transpose(transforms.ToTensor()(valid_dataset.data[0]),(1,0,2))
svhn_fig2 = svhn_fig1
svhn_fig4 = svhn_fig1
svhn_fig8 = svhn_fig1
svhn_fig16 = svhn_fig1
svhn_fig2 = transforms.Resize((16,16))(svhn_fig2)
svhn_fig4 = transforms.Resize((8,8))(svhn_fig4)
svhn_fig8 = transforms.Resize((4,4))(svhn_fig8)
svhn_fig16 = transforms.Resize((2,2))(svhn_fig16)

svhn_fig1 = np.transpose(transforms.Resize((32,32))(svhn_fig1),(2,1,0))
svhn_fig2 = np.transpose(transforms.Resize((32,32))(svhn_fig2),(2,1,0))
svhn_fig4 = np.transpose(transforms.Resize((32,32))(svhn_fig4),(2,1,0))
svhn_fig8 = np.transpose(transforms.Resize((32,32))(svhn_fig8),(2,1,0))
svhn_fig16 = np.transpose(transforms.Resize((32,32))(svhn_fig16),(2,1,0))

svhn_fig = plt.figure(dpi=300)
plt.subplot(1,5,1)
plt.axis('off')
plt.imshow(svhn_fig1)
plt.title(title1,fontsize=7)
plt.subplot(1,5,2)
plt.axis('off')
plt.imshow(svhn_fig2)
plt.title(title2,fontsize=7)
plt.subplot(1,5,3)
plt.axis('off')
plt.imshow(svhn_fig4)
plt.title(title3,fontsize=7)
plt.subplot(1,5,4)
plt.axis('off')
plt.imshow(svhn_fig8)
plt.title(title4,fontsize=7)
plt.subplot(1,5,5)
plt.axis('off')
plt.imshow(svhn_fig16)
plt.title(title5,fontsize=7)

plt.savefig('svhn_fig.svg',bbox_inches='tight')


valid_dataset = datasets.CIFAR10(root='cifar_data', 
                               train=False)


cifar_fig1 = transforms.ToTensor()(valid_dataset.data[0])
cifar_fig2 = cifar_fig1
cifar_fig4 = cifar_fig1
cifar_fig8 = cifar_fig1
cifar_fig16 = cifar_fig1
cifar_fig2 = transforms.Resize((16,16))(cifar_fig2)
cifar_fig4 = transforms.Resize((8,8))(cifar_fig4)
cifar_fig8 = transforms.Resize((4,4))(cifar_fig8)
cifar_fig16 = transforms.Resize((2,2))(cifar_fig16)

cifar_fig1 = np.transpose(transforms.Resize((32,32))(cifar_fig1),(1,2,0))
cifar_fig2 = np.transpose(transforms.Resize((32,32))(cifar_fig2),(1,2,0))
cifar_fig4 = np.transpose(transforms.Resize((32,32))(cifar_fig4),(1,2,0))
cifar_fig8 = np.transpose(transforms.Resize((32,32))(cifar_fig8),(1,2,0))
cifar_fig16 = np.transpose(transforms.Resize((32,32))(cifar_fig16),(1,2,0))

cifar_fig = plt.figure(dpi=300)
plt.subplot(1,5,1)
plt.axis('off')
plt.imshow(cifar_fig1)
plt.title(title1,fontsize=7)
plt.subplot(1,5,2)
plt.axis('off')
plt.imshow(cifar_fig2)
plt.title(title2,fontsize=7)
plt.subplot(1,5,3)
plt.axis('off')
plt.imshow(cifar_fig4)
plt.title(title3,fontsize=7)
plt.subplot(1,5,4)
plt.axis('off')
plt.imshow(cifar_fig8)
plt.title(title4,fontsize=7)
plt.subplot(1,5,5)
plt.axis('off')
plt.imshow(cifar_fig16)
plt.title(title5,fontsize=7)

plt.savefig('cifar_fig.svg',bbox_inches='tight')


title1 = "341x256"
title2 = "171x128"
title3 = "85x64"
title4 = "43x32"
title5 = "21x16"

imagenet_sample = Image.open("/Volumes/T7 Shield/Pretrained/ILSVRC2012_val_00000293.JPEG")
# imagenet_sample = Image.open("/Volumes/T7 Shield/Pretrained/ILSVRC2012_val_00000550.JPEG")

imagenet_fig256 = transforms.Resize(256)(transforms.Resize(256)(imagenet_sample))
imagenet_fig128 = transforms.Resize(256)(transforms.Resize(128)(imagenet_sample))
imagenet_fig64 = transforms.Resize(256)(transforms.Resize(64)(imagenet_sample))
imagenet_fig32 = transforms.Resize(256)(transforms.Resize(32)(imagenet_sample))
imagenet_fig16 = transforms.Resize(256)(transforms.Resize(16)(imagenet_sample))

imagenet_fig = plt.figure(dpi=300)
plt.subplot(1,5,1)
plt.axis('off')
plt.imshow(imagenet_fig256)
plt.title(title1,fontsize=7)
plt.subplot(1,5,2)
plt.axis('off')
plt.imshow(imagenet_fig128)
plt.title(title2,fontsize=7)
plt.subplot(1,5,3)
plt.axis('off')
plt.imshow(imagenet_fig64)
plt.title(title3,fontsize=7)
plt.subplot(1,5,4)
plt.axis('off')
plt.imshow(imagenet_fig32)
plt.title(title4,fontsize=7)
plt.subplot(1,5,5)
plt.axis('off')
plt.imshow(imagenet_fig16)
plt.title(title5,fontsize=7)

plt.savefig('imagenet_fig.svg',bbox_inches='tight')

title1 = "256x341"
title2 = "128x171"
title3 = "64x85"
title4 = "32x43"
title5 = "16x21"

imagenet2_sample = Image.open("/Volumes/T7 Shield/Pretrained/58fbc3e79ef15162b7726de04e98c90bb91a3055.jpeg")
imagenet2_sample = Image.open("/Volumes/T7 Shield/Pretrained/first_sleepingbag.png")

imagenet2_fig256 = transforms.Resize(256)(transforms.Resize(256)(imagenet2_sample))
imagenet2_fig128 = transforms.Resize(256)(transforms.Resize(128)(imagenet2_sample))
imagenet2_fig64 = transforms.Resize(256)(transforms.Resize(64)(imagenet2_sample))
imagenet2_fig32 = transforms.Resize(256)(transforms.Resize(32)(imagenet2_sample))
imagenet2_fig16 = transforms.Resize(256)(transforms.Resize(16)(imagenet2_sample))

imagenet2_fig = plt.figure(dpi=300)
plt.subplot(1,5,1)
plt.axis('off')
plt.imshow(imagenet2_fig256)
plt.title(title1,fontsize=7)
plt.subplot(1,5,2)
plt.axis('off')
plt.imshow(imagenet2_fig128)
plt.title(title2,fontsize=7)
plt.subplot(1,5,3)
plt.axis('off')
plt.imshow(imagenet2_fig64)
plt.title(title3,fontsize=7)
plt.subplot(1,5,4)
plt.axis('off')
plt.imshow(imagenet2_fig32)
plt.title(title4,fontsize=7)
plt.subplot(1,5,5)
plt.axis('off')
plt.imshow(imagenet2_fig16)
plt.title(title5,fontsize=7)

plt.savefig('imagenet2_fig.svg',bbox_inches='tight')
