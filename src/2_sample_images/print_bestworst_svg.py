import os
import numpy as np
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
from PIL import Image
from imagenetv2_pytorch import ImageNetV2Dataset

# Define an absolute path for the output directory
output_dir = os.path.abspath('../output/Figure3')

# Function to generate and save best and worst figures
def generate_and_save_bestworst_figures(dataset, best_indices, worst_indices, save_name, cmap=None, ratios=None):
    """
    dataset: The dataset to extract images from
    best_indices: List of indices for the best images
    worst_indices: List of indices for the worst images
    save_name: The name for saving the output SVG file
    output_dir: Absolute path for the output directory
    cmap: Colormap for grayscale images (use 'gray_r' for MNIST-like datasets, None for RGB)
    ratios: List of width ratios for ImageNet-like datasets (optional)
    """

    best1, best2 = dataset[best_indices[0]][0], dataset[best_indices[1]][0]
    worst1, worst2 = dataset[worst_indices[0]][0], dataset[worst_indices[1]][0]

    fig, axes = plt.subplots(1, 4, gridspec_kw={'width_ratios': ratios} if ratios else None, dpi=300)
    titles = ["High #1", "High #2", "Low #1", "Low #2"]

    for i, (image, title) in enumerate(zip([best1, best2, worst1, worst2], titles)):
        axes[i].imshow(image, cmap=cmap)
        axes[i].axis('off')
        axes[i].set_title(title, fontsize=7)

    os.makedirs(output_dir, exist_ok=True)  # Ensure the output directory exists
    output_path = os.path.join(output_dir, f'{save_name}.svg')
    plt.savefig(output_path, bbox_inches='tight')

# MNIST Dataset
mnist_dataset = datasets.MNIST(root='../data/mnist', train=False)
generate_and_save_bestworst_figures(mnist_dataset, [8413, 6657], [7846, 249], "mnist_bestworst", cmap='gray_r')

# KMNIST Dataset
kmnist_dataset = datasets.KMNIST(root='../data/kmnist', train=False)
generate_and_save_bestworst_figures(kmnist_dataset, [5184, 4151], [5174, 9683], "kmnist_bestworst", cmap='gray_r')

# FashionMNIST Dataset
fashionmnist_dataset = datasets.FashionMNIST(root='../data/fashionmnist', train=False)
generate_and_save_bestworst_figures(fashionmnist_dataset, [4341, 2157], [6774, 4144], "fashionmnist_bestworst", cmap='gray_r')

# SVHN Dataset
svhn_dataset = datasets.SVHN(root='../data/svhn', split="test")
generate_and_save_bestworst_figures(svhn_dataset, [21183, 18596], [24946, 9990], "svhn_bestworst")

# CIFAR-10 Dataset
cifar_dataset = datasets.CIFAR10(root='../data/cifar', train=False)
generate_and_save_bestworst_figures(cifar_dataset, [4014, 3625], [6427, 4443], "cifar_bestworst")

# ImageNet Dataset
imagenet_dataset = datasets.ImageNet(root='../data/imagenet', split='val')
generate_and_save_bestworst_figures(imagenet_dataset, [19612, 11009], [30765, 47853], "imagenet_bestworst", ratios=[700/469, 800/1200, 530/194, 380/500])

# ImageNetV2 Dataset
original_wd = os.getcwd()
os.chdir('../data/imagenetv2')
imagenetv2_dataset = ImageNetV2Dataset("matched-frequency")
generate_and_save_bestworst_figures(imagenetv2_dataset, [5139, 9889], [6944, 8465], "imagenetv2_bestworst")
os.chdir(original_wd)