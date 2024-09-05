# MNIST cleaning from https://pjreddie.com/projects/mnist-in-csv/

import pickle
import numpy as np
import pandas as pd
from scipy.io import loadmat

def convert_mnist(imgf, labelf, outf, n):
    f = open(imgf, "rb")
    o = open(outf, "w")
    l = open(labelf, "rb")

    f.read(16)
    l.read(8)
    images = []

    for i in range(n):
        image = [ord(l.read(1))]
        for j in range(28*28):
            image.append(ord(f.read(1)))
        images.append(image)

    for image in images:
        o.write(",".join(str(pix) for pix in image)+"\n")
    f.close()
    o.close()
    l.close()

datasets = ["MNIST", "KMNIST", "FashionMNIST"]

for dataset in datasets:
    dataset_lower = dataset.lower()
    convert_mnist(f"../data/{dataset_lower}/{dataset}/raw/train-images-idx3-ubyte", 
            f"../data/{dataset_lower}/{dataset}/raw/train-labels-idx1-ubyte",
            f"../data/{dataset_lower}/{dataset_lower}_train.csv", 60000)
    convert_mnist(f"../data/{dataset_lower}/{dataset}/raw/t10k-images-idx3-ubyte", 
            f"../data/{dataset_lower}/{dataset}/raw/t10k-labels-idx1-ubyte",
            f"../data/{dataset_lower}/{dataset_lower}_test.csv", 10000)


def load_cifar10_batch(file_path):
    with open(file_path, 'rb') as f:
        batch = pickle.load(f, encoding='latin1')
    data = batch['data']
    labels = batch['labels']
    return data, labels

# Directory containing CIFAR-10 batch files
data_dir = '../data/cifar/cifar-10-batches-py/'

# Load training data from batches 1-5
train_data = []
train_labels = []

for i in range(1, 6):
    data, labels = load_cifar10_batch(f'{data_dir}data_batch_{i}')
    train_data.append(data)
    train_labels.append(labels)

# Combine the training data and labels
train_data = np.vstack(train_data)
train_labels = np.hstack(train_labels)

# Combine data and labels into a single array for training
train_combined = np.column_stack((train_data, train_labels))

# Convert to DataFrame
train_df = pd.DataFrame(train_combined)

# Add column headers (optional)
pixel_columns = [f'pixel_{i}' for i in range(3072)]
train_df.columns = pixel_columns + ['label']

# Write the training DataFrame to a CSV file
train_df.to_csv('../data/cifar/cifar_train.csv', index=False)

# Load test data
test_data, test_labels = load_cifar10_batch(f'{data_dir}test_batch')

# Combine data and labels into a single array for testing
test_combined = np.column_stack((test_data, test_labels))

# Convert to DataFrame
test_df = pd.DataFrame(test_combined)

# Add column headers (optional)
test_df.columns = pixel_columns + ['label']

# Write the test DataFrame to a CSV file
test_df.to_csv('../data/cifar/cifar_test.csv', index=False)


def convert_svhn(sample):
    infile = f"../data/svhn/{sample}_32x32.mat"
    outfile = f"../data/svhn/svhn_{sample}.csv"
    svhn_data = loadmat(infile)
    # extract images and labels
    X = svhn_data['X']
    y = svhn_data['y']
    # Get the number of images
    num_images = X.shape[3]
    # Prepare a list to hold flattened image data and labels
    data = []
    # Loop over each image
    for i in range(num_images):
        # Flatten the image and convert it to a list
        flattened_image = X[:, :, :, i].flatten().tolist()
        # Append the label to the flattened image
        flattened_image.append(y[i][0])
        # Add to the data list
        data.append(flattened_image)
    # Convert the data list to a DataFrame
    df = pd.DataFrame(data)
    # Optionally, create headers for the DataFrame
    columns = [f'pixel_{i}' for i in range(X.shape[0] * X.shape[1] * X.shape[2])]
    columns.append('label')
    df.columns = columns
    # Write the DataFrame to a CSV file
    df.to_csv(outfile, index=False)

convert_svhn('train')
convert_svhn('test')




