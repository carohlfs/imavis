import numpy as np
from datetime import datetime 
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import argparse

parser = argparse.ArgumentParser(description='Training')
parser.add_argument('--dataset', choices=['mnist', 'kmnist', 'fashionmnist', 'cifar', 'svhn'], required=True, help='Dataset to use')
args = parser.parse_args()

# check device
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# parameters
RANDOM_SEED = 42
LEARNING_RATE = 0.001
BATCH_SIZE = 32
N_EPOCHS = 15

IMG_SIZE = 32
N_CLASSES = 10

if args.dataset == 'svhn':
    N_TRAIN = 73257
    N_VALID = 26032
elif args.dataset == 'cifar':
    N_TRAIN = 50000
    N_VALID = 10000
else:    
    N_TRAIN = 60000
    N_VALID = 10000


if args.dataset == 'cifar' or args.dataset == 'svhn':
    N_PIXELS = 3072
else:
    N_PIXELS = 1024

data_root = f"../data/{args.dataset}"

def get_accuracy(model, data_loader, device):
    '''
    Function for computing the accuracy of the predictions over the entire data_loader
    '''
    correct_pred = 0 
    n = 0
    with torch.no_grad():
        model.eval()
        for X, y_true in data_loader:
            X = X.to(device)
            y_true = y_true.to(device)
            _, y_prob = model(X)
            _, predicted_labels = torch.max(y_prob, 1)
            n += y_true.size(0)
            correct_pred += (predicted_labels == y_true).sum()
    return correct_pred.float() / n


def get_X_data(model, data_loader, device):
    '''
    Function for computing the accuracy of the predictions over the entire data_loader
    '''
    data_list = []
    with torch.no_grad():
        model.eval()
        for X, y_true in data_loader:
            data_list.append(X.to(device))
    return data_list


def get_probs(model, data_loader, device):
    '''
    Function for computing the accuracy of the predictions over the entire data_loader
    '''
    probabilities = []
    with torch.no_grad():
        model.eval()
        for X, y_true in data_loader:
            X = X.to(device)
            probabilities.append(y_true.to(device))
            _, y_prob = model(X)
            _, predicted_labels = torch.max(y_prob, 1)
            probabilities.append(y_prob)
    return probabilities


def plot_losses(train_losses, valid_losses):
    '''
    Function for plotting training and validation losses
    '''
    # temporarily change the style of the plots to seaborn 
    plt.style.use('seaborn-v0_8')
    train_losses = np.array(train_losses) 
    valid_losses = np.array(valid_losses)
    fig, ax = plt.subplots(figsize = (8, 4.5))
    ax.plot(train_losses, color='blue', label='Training loss') 
    ax.plot(valid_losses, color='red', label='Validation loss')
    ax.set(title="Loss over epochs", 
            xlabel='Epoch',
            ylabel='Loss') 
    ax.legend()
    fig.show()   
    # change the plot style to default
    plt.style.use('default')
    

def train(train_loader, model, criterion, optimizer, device):
    '''
    Function for the training step of the training loop
    '''
    model.train()
    running_loss = 0
    for X, y_true in train_loader:
        optimizer.zero_grad()
        X = X.to(device)
        y_true = y_true.to(device)
        # Forward pass
        y_hat, _ = model(X) 
        loss = criterion(y_hat, y_true) 
        running_loss += loss.item() * X.size(0)
        # Backward pass
        loss.backward()
        optimizer.step()        
    epoch_loss = running_loss / len(train_loader.dataset)
    return model, optimizer, epoch_loss


def validate(valid_loader, model, criterion, device):
    '''
    Function for the validation step of the training loop
    '''
    model.eval()
    running_loss = 0
    for X, y_true in valid_loader:  
        X = X.to(device)
        y_true = y_true.to(device)
        # Forward pass and record loss
        y_hat, _ = model(X) 
        loss = criterion(y_hat, y_true) 
        running_loss += loss.item() * X.size(0)
    epoch_loss = running_loss / len(valid_loader.dataset)
    return model, epoch_loss


def training_loop(model, criterion, optimizer, train_loader, valid_loader, epochs, device, print_every=1):
    '''
    Function defining the entire training loop
    '''
    # set objects for storing metrics
    best_loss = 1e10
    train_losses = []
    valid_losses = [] 
    # Train model
    for epoch in range(0, epochs):
        # training
        model, optimizer, train_loss = train(train_loader, model, criterion, optimizer, device)
        train_losses.append(train_loss)
        # validation
        with torch.no_grad():
            model, valid_loss = validate(valid_loader, model, criterion, device)
            valid_losses.append(valid_loss)
        if epoch % print_every == (print_every - 1):
            train_acc = get_accuracy(model, train_loader, device=device)
            valid_acc = get_accuracy(model, valid_loader, device=device)
            print(f'{datetime.now().time().replace(microsecond=0)} --- '
                  f'Epoch: {epoch}\t'
                  f'Train loss: {train_loss:.4f}\t'
                  f'Valid loss: {valid_loss:.4f}\t'
                  f'Train accuracy: {100 * train_acc:.2f}\t'
                  f'Valid accuracy: {100 * valid_acc:.2f}')
    plot_losses(train_losses, valid_losses)
    return model, optimizer, (train_losses, valid_losses)


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


# define transforms and configuration for different downsampling levels
# Note that "to tensor" happens first for the RGB datasets and
# last for the MNIST-style datasets.
if args.dataset == 'cifar' or args.dataset == 'svhn':
    # main transformation
    transforms_main = transforms.Compose([
        transforms.ToTensor(),
        transforms.Grayscale(),
        transforms.Resize((32, 32))
        ])
    # possible downsampling rates
    downsampling_configs = {
        'ds1': 32,
        'ds2': 16,
        'ds4': 8,
        'ds8': 4,
        'ds16': 2
        }
    # Function to create transformations
    def create_transform(downsampling_size, grayscale=False):
        transform_list = [transforms.ToTensor()]
        if grayscale:
            transform_list.append(transforms.Grayscale())
        transform_list.append(transforms.Resize((downsampling_size, downsampling_size)))
        transform_list.append(transforms.Resize((32, 32)))
        return transforms.Compose(transform_list)

else:
    # main transformation
    transforms_main = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        ])
    # possible downsampling rates
    downsampling_configs = {
        'ds1': 28,
        'ds2': 14,
        'ds4': 7,
        'ds7': 4,
        'ds14': 2
        }
    # Function to create transformations
    def create_transform(downsampling_size, grayscale=False):
        transform_list = [transforms.Resize((downsampling_size, downsampling_size))]
        transform_list.append(transforms.Resize((32, 32)))
        transform_list.append(transforms.ToTensor())
        return transforms.Compose(transform_list)


# download and create datasets
if args.dataset == 'mnist':
    train_dataset = datasets.MNIST(root=data_root,train=True,transform=transforms_main,download=True)
    valid_dataset = datasets.MNIST(root=data_root,train=False,transform=transforms_main)
elif args.dataset == 'kmnist':
    train_dataset = datasets.KMNIST(root=data_root,train=True,transform=transforms_main,download=True)
    valid_dataset = datasets.KMNIST(root=data_root,train=False,transform=transforms_main)
elif args.dataset == 'fashionmnist':
    train_dataset = datasets.FashionMNIST(root=data_root,train=True,transform=transforms_main,download=True)
    valid_dataset = datasets.FashionMNIST(root=data_root,train=False,transform=transforms_main)
elif args.dataset == 'cifar':
    # incorporate horizontal flipping only into CIFAR training
    # because of their horizontal symmetry.
    transforms_train = transforms.Compose([
        transforms.ToTensor(),
        transforms.Grayscale(),
        transforms.RandomHorizontalFlip(),
        transforms.Resize((32, 32))
    ])
    train_dataset = datasets.CIFAR10(root=data_root,train=True,transform=transforms_train,download=True)
    valid_dataset = datasets.CIFAR10(root=data_root,train=False,transform=transforms_main,download=True)
elif args.dataset == 'svhn':
    train_dataset = datasets.SVHN(root=data_root,split="train",transform=transforms_main,download=True)
    valid_dataset = datasets.SVHN(root=data_root,split="test",transform=transforms_main,download=True)


# define the data loaders
train_loader = DataLoader(dataset=train_dataset,batch_size=BATCH_SIZE,shuffle=False)
valid_loader = DataLoader(dataset=valid_dataset,batch_size=BATCH_SIZE,shuffle=False)

torch.manual_seed(RANDOM_SEED)

model = LeNet5(N_CLASSES).to(DEVICE)
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
criterion = nn.CrossEntropyLoss()

model, optimizer, _ = training_loop(model, criterion, optimizer, train_loader, valid_loader, N_EPOCHS, DEVICE)

# Function to save probabilities and classes
def save_probs_and_classes(model, loader, device, filename_prefix, size):
    probs = get_probs(model, loader, device)
    if size in [28, 32]:
        classes = probs[0].detach().numpy()
        np.savetxt(f"{filename_prefix}_preds.csv", classes, delimiter=",")
    probabilities = probs[1].detach().numpy()
    np.savetxt(f"{filename_prefix}_probs.csv", probabilities, delimiter=",")

# Function to save X data
def save_X_data(model, loader, device, filename_prefix, num_samples):
    Xmat = get_X_data(model, loader, device)
    Xmat = Xmat[0].detach().numpy().reshape(num_samples, N_PIXELS)
    np.savetxt(f"{filename_prefix}_X.csv", Xmat, delimiter=",")

# for CIFAR and SVHN, save Xes in RGB, but MNIST-style are in B/W
# all use B/W for the model runs with the probabilities & classes
if args.dataset == 'cifar':
    transform = create_transform(32)
    train_dataset = datasets.CIFAR10(root=data_root,train=True,transform=transform)
    # increase loader batch size to N_TRAIN in order to grab all observations in functions.
    train_loader = DataLoader(dataset=train_dataset, batch_size=N_TRAIN, shuffle=False)
    save_X_data(model, train_loader, DEVICE, f"../tmp/probs/{args.dataset}{downsampling_configs['ds1']}_train", N_TRAIN)
    transform = create_transform(32,grayscale=True)
    train_dataset = datasets.CIFAR10(root=data_root,train=True,transform=transform)
    train_loader = DataLoader(dataset=train_dataset, batch_size=N_TRAIN, shuffle=False)
elif args.dataset == 'svhn':
    transform = create_transform(32)
    train_dataset = datasets.SVHN(root=data_root,split="train",transform=transform)
    # increase loader batch size to N_TRAIN in order to grab all observations in functions.
    train_loader = DataLoader(dataset=train_dataset, batch_size=N_TRAIN, shuffle=False)
    save_X_data(model, train_loader, DEVICE, f"../tmp/probs/{args.dataset}{downsampling_configs['ds1']}_train", N_TRAIN)
    transform = create_transform(32,grayscale=True)
    train_dataset = datasets.SVHN(root=data_root,split="train",transform=transform)
    train_loader = DataLoader(dataset=train_dataset, batch_size=N_TRAIN, shuffle=False)
else:
    train_loader = DataLoader(dataset=train_dataset, batch_size=N_TRAIN, shuffle=False)
    save_X_data(model, train_loader, DEVICE, f"../tmp/probs/{args.dataset}{downsampling_configs['ds1']}_train", N_TRAIN)

save_probs_and_classes(model, train_loader, DEVICE, f"../tmp/probs/{args.dataset}{downsampling_configs['ds1']}_train",downsampling_configs['ds1'])

# Main processing loop
for name, size in downsampling_configs.items():
    transform = create_transform(size)
    if args.dataset == 'mnist':
        valid_dataset = datasets.MNIST(root=data_root,train=False,transform=transform)
    elif args.dataset == 'kmnist':
        valid_dataset = datasets.KMNIST(root=data_root,train=False,transform=transform)
    elif args.dataset == 'fashionmnist':
        valid_dataset = datasets.FashionMNIST(root=data_root,train=False,transform=transform)        
    elif args.dataset == 'cifar':
        valid_dataset = datasets.CIFAR10(root=data_root,train=False,transform=transform) 
    elif args.dataset == 'svhn':
        valid_dataset = datasets.SVHN(root=data_root,split="test",transform=transform)   
    valid_loader = DataLoader(dataset=valid_dataset, batch_size=N_VALID, shuffle=False)
    save_X_data(model, valid_loader, DEVICE, f"../tmp/probs/{args.dataset}{size}_test", N_VALID)
    if args.dataset == 'cifar' or args.dataset=='svhn':
        transform = create_transform(size, grayscale=True)
        if args.dataset == 'cifar':
            valid_dataset = datasets.CIFAR10(root=data_root,train=False,transform=transform)
        else:
            valid_dataset = datasets.SVHN(root=data_root,split="test",transform=transform)
        valid_loader = DataLoader(dataset=valid_dataset, batch_size=N_VALID, shuffle=False)
    save_probs_and_classes(model, valid_loader, DEVICE, f"../tmp/probs/{args.dataset}{size}_test",size)
