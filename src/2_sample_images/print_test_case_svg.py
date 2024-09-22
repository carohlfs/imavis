import numpy as np
import torch
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
from PIL import Image

# Function to generate and save figures, with inferred settings based on save_name
def generate_and_save_figures(input_data, save_name):
    """
    input_data: Can be a dataset or a path to an image file
    save_name: The name for saving the output SVG file (used to infer image type)
    """

    # Infer dataset type from save_name and set sizes, titles, and transformations accordingly
    if "mnist" in save_name or "kmnist" in save_name or "fashionmnist" in save_name:
        sizes = [28, 14, 7, 4, 2]
        titles = ["28x28", "14x14", "7x7", "4x4", "2x2"]
        transform = transforms.Compose([
            transforms.Grayscale(),  # Convert to grayscale
            transforms.Resize((32, 32))  # Final resize to 32x32
        ])
        is_image_path = False
        is_grayscale = True

    elif "svhn" in save_name or "cifar" in save_name:
        sizes = [32, 16, 8, 4, 2]
        titles = ["32x32", "16x16", "8x8", "4x4", "2x2"]
        transform = transforms.Resize((32, 32))  # Resize first
        is_image_path = False
        is_grayscale = False

    elif "imagenet" in save_name:
        sizes = [256, 128, 64, 32, 16]
        titles = ["341x256", "171x128", "85x64", "43x32", "21x16"]
        transform = transforms.Resize(256)  # Directly resize to 256
        is_image_path = True
        is_grayscale = False

    fig = plt.figure(dpi=300)

    # Load the image or dataset accordingly
    if is_image_path:
        image = Image.open(input_data)  # Load image from path for ImageNet
    else:
        if hasattr(input_data, 'data'):  # Handle MNIST-like datasets
            if isinstance(input_data.data, torch.Tensor):
                image = input_data.data[0].numpy()  # Convert to NumPy if tensor
            else:
                image = input_data.data[0]
        else:  # Handle SVHN and CIFAR-like datasets
            image = input_data[0]  # Access first image

        # CIFAR-10 and SVHN images need to be converted to PIL Image with RGB format
        if not is_grayscale:
            if len(image.shape) == 3 and image.shape[0] == 3:  # Check if RGB format
                image = np.transpose(image, (1, 2, 0))  # Transpose to HWC for PIL

    for i, size in enumerate(sizes):
        # Handle different image formats
        if is_grayscale:  # Grayscale images
            img_resized = transform(transforms.Resize(size)(Image.fromarray(image)))
        else:  # RGB images
            img_resized = transform(transforms.Resize(size)(Image.fromarray(np.array(image).astype(np.uint8))))

        # Plot the resized image
        plt.subplot(1, len(sizes), i + 1)
        plt.axis('off')
        plt.imshow(img_resized, cmap='gray_r' if is_grayscale else None)
        plt.title(titles[i], fontsize=7)

    plt.savefig(f'../output/Figure2/{save_name}.svg', bbox_inches='tight')

# MNIST Dataset
mnist_dataset = datasets.MNIST(root='../data/mnist', train=False)
generate_and_save_figures(mnist_dataset, "mnist_fig")

# KMNIST Dataset
kmnist_dataset = datasets.KMNIST(root='../data/kmnist', train=False)
generate_and_save_figures(kmnist_dataset, "kmnist_fig")

# FashionMNIST Dataset
fashionmnist_dataset = datasets.FashionMNIST(root='../data/fashionmnist', train=False)
generate_and_save_figures(fashionmnist_dataset, "fashionmnist_fig")

# SVHN Dataset
svhn_dataset = datasets.SVHN(root='../data/svhn', split="test")
generate_and_save_figures(svhn_dataset, "svhn_fig")

# CIFAR10 Dataset
cifar_dataset = datasets.CIFAR10(root='../data/cifar', train=False)
generate_and_save_figures(cifar_dataset, "cifar_fig")

# For ImageNet and ImageNet-V2, reference the first validation or test
# image directly by name to ensure a match with the ones used in the paper.

# for ImageNet, it's the very first tench (class 0) image.
# ImageNet Dataset
imagenet_sample_path = "../data/imagenet/val/n01440764/ILSVRC2012_val_00000293.JPEG"
generate_and_save_figures(imagenet_sample_path, "imagenet_fig")

# for ImageNet-V2, it's a sleeping bag (category 797), which happens to be
# the first element of the dataset when it's loaded via PyTorch.

# ImageNetV2 Dataset
imagenetv2_sample_path = "../data/imagenetv2/ImageNetV2-matched-frequency/797/32bef66d872325ebe77fb9242d5eee4e7afabe66.jpeg"
generate_and_save_figures(imagenetv2_sample_path, "imagenet2_fig")
