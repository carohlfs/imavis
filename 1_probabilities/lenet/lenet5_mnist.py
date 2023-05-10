# Eryk Lewinson
# erykml 
# https://towardsdatascience.com/implementing-yann-lecuns-lenet-5-in-pytorch-5e05a0911320
# https://github.com/erykml

import numpy as np
from datetime import datetime 
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from torchvision import datasets, transforms

import matplotlib.pyplot as plt

# check device
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# parameters
RANDOM_SEED = 42
LEARNING_RATE = 0.001
BATCH_SIZE = 32
N_EPOCHS = 15

IMG_SIZE = 32
N_CLASSES = 10

N_TRAIN = 60000
N_VALID = 10000

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
    plt.style.use('seaborn')
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



# define transforms
transforms_main = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
])


# download and create datasets
train_dataset = datasets.MNIST(root='mnist_data', 
                               train=True, 
                               transform=transforms_main,
                               download=True)


valid_dataset = datasets.MNIST(root='mnist_data', 
                               train=False, 
                               transform=transforms_main)


# define the data loaders
train_loader = DataLoader(dataset=train_dataset, 
                          batch_size=BATCH_SIZE, 
                          shuffle=False)


valid_loader = DataLoader(dataset=valid_dataset, 
                          batch_size=BATCH_SIZE, 
                          shuffle=False)


torch.manual_seed(RANDOM_SEED)

model = LeNet5(N_CLASSES).to(DEVICE)
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
criterion = nn.CrossEntropyLoss()

model, optimizer, _ = training_loop(model, criterion, optimizer, train_loader, valid_loader, N_EPOCHS, DEVICE)

transforms_ds2 = transforms.Compose([
    transforms.Resize((14, 14)),
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
])

transforms_ds4 = transforms.Compose([
    transforms.Resize((7, 7)),
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
])

transforms_ds7 = transforms.Compose([
    transforms.Resize((4, 4)),
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
])

transforms_ds14 = transforms.Compose([
    transforms.Resize((2, 2)),
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
])

train_loader = DataLoader(dataset=train_dataset, 
                          batch_size=N_TRAIN, 
                          shuffle=False)

probs = get_probs(model, train_loader, DEVICE)
classes = probs[0].detach().numpy()
np.savetxt("mnist_train_actuals.csv",classes,delimiter=",")
probs = probs[1].detach().numpy()
np.savetxt("mnist_train_probs.csv",probs,delimiter=",")
Xmat = get_X_data(model, train_loader, DEVICE)
Xmat = Xmat[0].detach().numpy().reshape(N_TRAIN,1024)
np.savetxt("mnist1_train_X.csv",Xmat,delimiter=",")

valid_loader = DataLoader(dataset=valid_dataset, 
                          batch_size=N_VALID, 
                          shuffle=False)


probs = get_probs(model, valid_loader, DEVICE)
classes = probs[0].detach().numpy()
np.savetxt("mnist_test_actuals.csv",classes,delimiter=",")
probs = probs[1].detach().numpy()
np.savetxt("mnist_test_probs.csv",probs,delimiter=",")
Xmat = get_X_data(model, valid_loader, DEVICE)
Xmat = Xmat[0].detach().numpy().reshape(N_VALID,1024)
np.savetxt("mnist1_test_X.csv",Xmat,delimiter=",")

train_dataset = datasets.MNIST(root='mnist_data', 
                               train=True, 
                               transform=transforms_ds2)


train_loader = DataLoader(dataset=train_dataset, 
                          batch_size=N_TRAIN, 
                          shuffle=False)


probs = get_probs(model, train_loader, DEVICE)
probs = probs[1].detach().numpy()
np.savetxt("mnist2_train_probs.csv",probs,delimiter=",")
Xmat = get_X_data(model, train_loader, DEVICE)
Xmat = Xmat[0].detach().numpy().reshape(N_TRAIN,1024)
np.savetxt("mnist2_train_X.csv",Xmat,delimiter=",")

valid_dataset = datasets.MNIST(root='mnist_data', 
                               train=False, 
                               transform=transforms_ds2)


valid_loader = DataLoader(dataset=valid_dataset, 
                          batch_size=N_VALID, 
                          shuffle=False)


probs = get_probs(model, valid_loader, DEVICE)
probs = probs[1].detach().numpy()
np.savetxt("mnist2_test_probs.csv",probs,delimiter=",")
Xmat = get_X_data(model, valid_loader, DEVICE)
Xmat = Xmat[0].detach().numpy().reshape(N_VALID,1024)
np.savetxt("mnist2_test_X.csv",Xmat,delimiter=",")

train_dataset = datasets.MNIST(root='mnist_data', 
                               train=True, 
                               transform=transforms_ds4)


train_loader = DataLoader(dataset=train_dataset, 
                          batch_size=N_TRAIN, 
                          shuffle=False)


probs = get_probs(model, train_loader, DEVICE)
probs = probs[1].detach().numpy()
np.savetxt("mnist4_train_probs.csv",probs,delimiter=",")
Xmat = get_X_data(model, train_loader, DEVICE)
Xmat = Xmat[0].detach().numpy().reshape(N_TRAIN,1024)
np.savetxt("mnist4_train_X.csv",Xmat,delimiter=",")

valid_dataset = datasets.MNIST(root='mnist_data', 
                               train=False, 
                               transform=transforms_ds4)


valid_loader = DataLoader(dataset=valid_dataset, 
                          batch_size=N_VALID, 
                          shuffle=False)


probs = get_probs(model, valid_loader, DEVICE)
probs = probs[1].detach().numpy()
np.savetxt("mnist4_test_probs.csv",probs,delimiter=",")
Xmat = get_X_data(model, valid_loader, DEVICE)
Xmat = Xmat[0].detach().numpy().reshape(N_VALID,1024)
np.savetxt("mnist4_test_X.csv",Xmat,delimiter=",")

train_dataset = datasets.MNIST(root='mnist_data', 
                               train=True, 
                               transform=transforms_ds7)


train_loader = DataLoader(dataset=train_dataset, 
                          batch_size=N_TRAIN, 
                          shuffle=False)


probs = get_probs(model, train_loader, DEVICE)
probs = probs[1].detach().numpy()
np.savetxt("mnist7_train_probs.csv",probs,delimiter=",")
Xmat = get_X_data(model, train_loader, DEVICE)
Xmat = Xmat[0].detach().numpy().reshape(N_TRAIN,1024)
np.savetxt("mnist7_train_X.csv",Xmat,delimiter=",")

valid_dataset = datasets.MNIST(root='mnist_data', 
                               train=False, 
                               transform=transforms_ds7)


valid_loader = DataLoader(dataset=valid_dataset, 
                          batch_size=N_VALID, 
                          shuffle=False)


probs = get_probs(model, valid_loader, DEVICE)
probs = probs[1].detach().numpy()
Xmat = get_X_data(model, valid_loader, DEVICE)
Xmat = Xmat[0].detach().numpy().reshape(N_VALID,1024)
np.savetxt("mnist7_test_probs.csv",probs,delimiter=",")
np.savetxt("mnist7_test_X.csv",Xmat,delimiter=",")

train_dataset = datasets.MNIST(root='mnist_data', 
                               train=True, 
                               transform=transforms_ds14)


train_loader = DataLoader(dataset=train_dataset, 
                          batch_size=N_TRAIN, 
                          shuffle=False)


probs = get_probs(model, train_loader, DEVICE)
probs = probs[1].detach().numpy()
np.savetxt("mnist14_train_probs.csv",probs,delimiter=",")
Xmat = get_X_data(model, train_loader, DEVICE)
Xmat = Xmat[0].detach().numpy().reshape(N_TRAIN,1024)
np.savetxt("mnist14_train_X.csv",Xmat,delimiter=",")

valid_dataset = datasets.MNIST(root='mnist_data', 
                               train=False, 
                               transform=transforms_ds14)


valid_loader = DataLoader(dataset=valid_dataset, 
                          batch_size=N_VALID, 
                          shuffle=False)


probs = get_probs(model, valid_loader, DEVICE)
probs = probs[1].detach().numpy()
np.savetxt("mnist14_test_probs.csv",probs,delimiter=",")
Xmat = get_X_data(model, valid_loader, DEVICE)
Xmat = Xmat[0].detach().numpy().reshape(N_VALID,1024)
np.savetxt("mnist14_test_X.csv",Xmat,delimiter=",")


