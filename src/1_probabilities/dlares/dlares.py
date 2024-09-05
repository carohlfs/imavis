import sys
import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torchvision
import torchvision.transforms as transforms
import argparse

# Add the src directory to the system path
sys.path.append(os.path.abspath('../src/1_probabilities/dlares'))

# Import your models and utility functions
from dla_simple import SimpleDLA
from resnet import ResNet
from utils import progress_bar

# Argument parser setup
parser = argparse.ArgumentParser(description='Training')
parser.add_argument('--dataset', choices=['cifar', 'svhn'], required=True, help='Dataset to use: cifar or svhn')
parser.add_argument('--model', choices=['dla', 'res'], required=True, help='Model to use: dla or res')
parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
args = parser.parse_args()

# Define arg-dependent filenames & directories
checkpoint_path = f"../tmp/checkpoint/{args.dataset}_{args.model}_ckpt.pth"
data_root = f"../data/{args.dataset}"

# Set device
device = 'cuda' if torch.cuda.is_available() else 'cpu'
best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch

# Data preparation
print('==> Preparing data..')

# Define common transforms
transform_train = transforms.Compose([
    transforms.ToTensor(),
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
])

# Set the data_root and classes based on the dataset
if args.dataset == 'cifar':
    N_TRAIN = 50000
    N_VALID = 10000
    normalization = transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    transform_train.transforms.append(normalization)
    trainset = torchvision.datasets.CIFAR10(
        root=data_root, train=True, download=True, transform=transform_train)
    transform_test.transforms.append(normalization)
    testset = torchvision.datasets.CIFAR10(
        root=data_root, train=False, download=True, transform=transform_test)

elif args.dataset == 'svhn':
    N_TRAIN = 73257
    N_VALID = 26032
    normalization = transforms.Normalize((0.4377, 0.4438, 0.4728), (0.1980, 0.2010, 0.1970))
    classes = ('1', '2', '3', '4', '5', '6', '7', '8', '9', '10')
    transform_train.transforms.append(normalization)
    trainset = torchvision.datasets.SVHN(
        root=data_root, split='train', download=True, transform=transform_train)
    transform_test.transforms.append(normalization)
    testset = torchvision.datasets.SVHN(
        root=data_root, split='test', download=True, transform=transform_test)

trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True, num_workers=2)
testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=2)

# Model
print('==> Building model..')
if args.model == 'res':
    net = ResNet18()

elif args.model == 'dla':
    net = SimpleDLA()


net = net.to(device)
if device == 'cuda':
    net = torch.nn.DataParallel(net)
    cudnn.benchmark = True

if args.resume:
    # Load checkpoint.
    print('==> Resuming from checkpoint..')
    assert os.path.isdir('../tmp/checkpoint'), 'Error: no checkpoint directory found!'
    checkpoint = torch.load(checkpoint_path)
    net.load_state_dict(checkpoint['net'])
    best_acc = checkpoint['acc']
    start_epoch = checkpoint['epoch']

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=args.lr,
                      momentum=0.9, weight_decay=5e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)

# Training
def train(epoch):
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
        progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                     % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))


def test(epoch):
    global best_acc
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)
            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                         % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))
    # Save checkpoint.
    acc = 100.*correct/total
    if acc > best_acc:
        print('Saving..')
        state = {
            'net': net.state_dict(),
            'acc': acc,
            'epoch': epoch,
        }
        if not os.path.isdir('../tmp/checkpoint'):
            os.mkdir('../tmp/checkpoint')
        torch.save(state, checkpoint_path)
        best_acc = acc


for epoch in range(start_epoch, start_epoch+200):
    train(epoch)
    test(epoch)
    scheduler.step()

checkpoint = torch.load(checkpoint_path)
net.load_state_dict(checkpoint['net'])
best_acc = checkpoint['acc']
start_epoch = checkpoint['epoch']
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=args.lr,
                      momentum=0.9, weight_decay=5e-4)


# Dictionary to hold transformation configurations
transform_configs = {
    'ds1': 32,
    'ds2': 16,
    'ds4': 8,
    'ds8': 4,
    'ds16': 2
}

# Now, for outputs, we switch to a common transform approach for train & test.

# Function to create transformations
def create_transform(resize):
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((resize, resize)),
        transforms.Resize((32, 32)),
        normalization
    ])

def get_probs(loader):
    probabilities = []
    net.eval()
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(loader):
            probabilities.append(targets.to(device))
            probabilities.append(net(inputs))
    return probabilities

# Function to save probabilities and classes
def save_probs_and_classes(prefix, loader):
    probs = get_probs(loader)
    classes = probs[0].detach().numpy()
    np.savetxt(f"../tmp/probs/{prefix}_{args.model}preds.csv", classes, delimiter=",")
    probabilities = probs[1].detach().numpy()
    np.savetxt(f"../tmp/probs/{prefix}_{args.model}probs.csv", probabilities, delimiter=",")

# Standard train and test transforms
transform_standard = create_transform(32)

if args.dataset == 'cifar':
    trainset = torchvision.datasets.CIFAR10(root=data_root, train=True, download=True, transform=transform_standard)
    testset = torchvision.datasets.CIFAR10(root=data_root, train=False, download=True, transform=transform_standard)

elif args.dataset == 'svhn':
    trainset = torchvision.datasets.SVHN(root=data_root, split='train', download=True, transform=transform_standard)
    testset = torchvision.datasets.SVHN(root=data_root, split='test', download=True, transform=transform_standard)

# Load and process the standard dataset
trainloader = torch.utils.data.DataLoader(trainset, batch_size=N_TRAIN, shuffle=False, num_workers=1)
testloader = torch.utils.data.DataLoader(testset, batch_size=N_VALID, shuffle=False, num_workers=1)

# Save standard dataset results for training sample
save_probs_and_classes(f"{args.dataset}_train32", trainloader)

# Loop through each downsample configuration
for name, resize in transform_configs.items():
    transform = create_transform(resize)
    # Load datasets with the current transformation
    if args.dataset == 'cifar':
        testset = torchvision.datasets.CIFAR10(root=cifar_root, train=False, download=True, transform=transform)
    elif args.dataset == 'svhn':
        testset = torchvision.datasets.SVHN(root=data_root, split='test', download=True, transform=transform)        
    testloader = torch.utils.data.DataLoader(testset, batch_size=N_VALID, shuffle=False, num_workers=1)
    # Save probabilities and classes for the current transformation
    save_probs_and_classes(f"{args.dataset}_test{resize}", testloader)
