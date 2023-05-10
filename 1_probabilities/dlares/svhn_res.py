'''Train SVHN with PyTorch.'''
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms

import os
import argparse

from models import *
from utils import progress_bar

N_TRAIN = 73257
N_VALID = 26032

parser = argparse.ArgumentParser(description='PyTorch SVHN Training')
parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
parser.add_argument('--resume', '-r', action='store_true',
                    help='resume from checkpoint')
args = parser.parse_args()

device = 'cuda' if torch.cuda.is_available() else 'cpu'
best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch

# Data
print('==> Preparing data..')
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.Normalize((0.4376821, 0.4437697, 0.47280442), (0.19803012, 0.20101562, 0.19703614)),
])

transform_test = transforms.Compose([
    transforms.Normalize((0.4376821, 0.4437697, 0.47280442), (0.19803012, 0.20101562, 0.19703614)),
])

trainset = torchvision.datasets.SVHN(
    root='./svhndata', split='train', download=True, transform=transform_train)
trainloader = torch.utils.data.DataLoader(
    trainset, batch_size=128, shuffle=True, num_workers=2)

testset = torchvision.datasets.SVHN(
    root='./svhndata', split='test', download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(
    testset, batch_size=100, shuffle=False, num_workers=2)

classes = ('1', '2', '3', '4', '5',
           '6', '7', '8', '9', '10')

# Model
print('==> Building model..')
# net = VGG('VGG19')
net = ResNet18()
# net = PreActResNet18()
# net = GoogLeNet()
# net = DenseNet121()
# net = ResNeXt29_2x64d()
# net = MobileNet()
# net = MobileNetV2()
# net = DPN92()
# net = ShuffleNetG2()
# net = SENet18()
# net = ShuffleNetV2(1)
# net = EfficientNetB0()
# net = RegNetX_200MF()
# net = SimpleDLA()
net = net.to(device)
if device == 'cuda':
    net = torch.nn.DataParallel(net)
    cudnn.benchmark = True

if args.resume:
    # Load checkpoint.
    print('==> Resuming from checkpoint..')
    assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
    checkpoint = torch.load('./checkpoint/ckpt.pth')
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
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        torch.save(state, './checkpoint/ckpt.pth')
        best_acc = acc


for epoch in range(start_epoch, start_epoch+200):
    train(epoch)
    test(epoch)
    scheduler.step()


transform_ds1 = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.Resize((32, 32)),
    transforms.Normalize((0.4376821, 0.4437697, 0.47280442), (0.19803012, 0.20101562, 0.19703614))
])

transform_ds2 = transforms.Compose([
    transforms.Resize((16, 16)),
    transforms.Resize((32, 32)),
    transforms.Normalize((0.4376821, 0.4437697, 0.47280442), (0.19803012, 0.20101562, 0.19703614))
])

transform_ds4 = transforms.Compose([
    transforms.Resize((8, 8)),
    transforms.Resize((32, 32)),
    transforms.Normalize((0.4376821, 0.4437697, 0.47280442), (0.19803012, 0.20101562, 0.19703614))
])

transform_ds8 = transforms.Compose([
    transforms.Resize((4, 4)),
    transforms.Resize((32, 32)),
    transforms.Normalize((0.4376821, 0.4437697, 0.47280442), (0.19803012, 0.20101562, 0.19703614))
])

transform_ds16 = transforms.Compose([
    transforms.Resize((2, 2)),
    transforms.Resize((32, 32)),
    transforms.Normalize((0.4376821, 0.4437697, 0.47280442), (0.19803012, 0.20101562, 0.19703614))
])

trainset = torchvision.datasets.SVHN(
    root='./svhndata', split='train', download=True, transform=transform_test)

trainloader = torch.utils.data.DataLoader(
    trainset, batch_size=N_TRAIN, shuffle=False, num_workers=2)

testset = torchvision.datasets.SVHN(
    root='./svhndata', split='test', download=True, transform=transform_test)

testloader = torch.utils.data.DataLoader(
    testset, batch_size=N_VALID, shuffle=False, num_workers=1)

import numpy as np

def get_train_probs():
    probabilities = []
    net.eval()
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(trainloader):
            probabilities.append(targets.to(device))
            probabilities.append(net(inputs))
    return probabilities


def get_test_probs():
    probabilities = []
    net.eval()
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            probabilities.append(targets.to(device))
            probabilities.append(net(inputs))
    return probabilities


probs = get_train_probs()
classes = probs[0].detach().numpy()
np.savetxt("svhn_train_dlaactuals.csv",classes,delimiter=",")
probs = probs[1].detach().numpy()
np.savetxt("svhn_train_dlaprobs.csv",probs,delimiter=",")

probs = get_test_probs()
classes = probs[0].detach().numpy()
np.savetxt("svhn_test_dlaactuals.csv",classes,delimiter=",")
probs = probs[1].detach().numpy()
np.savetxt("svhn_test_dlaprobs.csv",probs,delimiter=",")

trainset = torchvision.datasets.SVHN(
    root='./svhndata', split='train', download=True, transform=transform_ds1)

trainloader = torch.utils.data.DataLoader(
    trainset, batch_size=N_TRAIN, shuffle=False, num_workers=1)

testset = torchvision.datasets.SVHN(
    root='./svhndata', split='test', download=True, transform=transform_ds1)

testloader = torch.utils.data.DataLoader(
    testset, batch_size=N_VALID, shuffle=False, num_workers=1)

probs = get_train_probs()
classes = probs[0].detach().numpy()
np.savetxt("svhn1_train_dlaactuals.csv",classes,delimiter=",")
probs = probs[1].detach().numpy()
np.savetxt("svhn1_train_dlaprobs.csv",probs,delimiter=",")

probs = get_test_probs()
classes = probs[0].detach().numpy()
np.savetxt("svhn1_test_dlaactuals.csv",classes,delimiter=",")
probs = probs[1].detach().numpy()
np.savetxt("svhn1_test_dlaprobs.csv",probs,delimiter=",")

trainset = torchvision.datasets.SVHN(
    root='./svhndata', split='train', download=True, transform=transform_ds2)

trainloader = torch.utils.data.DataLoader(
    trainset, batch_size=N_TRAIN, shuffle=False, num_workers=1)

testset = torchvision.datasets.SVHN(
    root='./svhndata', split='test', download=True, transform=transform_ds2)

testloader = torch.utils.data.DataLoader(
    testset, batch_size=N_VALID, shuffle=False, num_workers=1)

probs = get_train_probs()
classes = probs[0].detach().numpy()
np.savetxt("svhn2_train_dlaactuals.csv",classes,delimiter=",")
probs = probs[1].detach().numpy()
np.savetxt("svhn2_train_dlaprobs.csv",probs,delimiter=",")

probs = get_test_probs()
classes = probs[0].detach().numpy()
np.savetxt("svhn2_test_dlaactuals.csv",classes,delimiter=",")
probs = probs[1].detach().numpy()
np.savetxt("svhn2_test_dlaprobs.csv",probs,delimiter=",")

trainset = torchvision.datasets.SVHN(
    root='./svhndata', split='train', download=True, transform=transform_ds4)

trainloader = torch.utils.data.DataLoader(
    trainset, batch_size=N_TRAIN, shuffle=False, num_workers=1)

testset = torchvision.datasets.SVHN(
    root='./svhndata', split='test', download=True, transform=transform_ds4)

testloader = torch.utils.data.DataLoader(
    testset, batch_size=N_VALID, shuffle=False, num_workers=1)

probs = get_train_probs()
classes = probs[0].detach().numpy()
np.savetxt("svhn4_train_dlaactuals.csv",classes,delimiter=",")
probs = probs[1].detach().numpy()
np.savetxt("svhn4_train_dlaprobs.csv",probs,delimiter=",")

probs = get_test_probs()
classes = probs[0].detach().numpy()
np.savetxt("svhn4_test_dlaactuals.csv",classes,delimiter=",")
probs = probs[1].detach().numpy()
np.savetxt("svhn4_test_dlaprobs.csv",probs,delimiter=",")

trainset = torchvision.datasets.SVHN(
    root='./svhndata', split='train', download=True, transform=transform_ds8)

trainloader = torch.utils.data.DataLoader(
    trainset, batch_size=N_TRAIN, shuffle=False, num_workers=1)

testset = torchvision.datasets.SVHN(
    root='./svhndata', split='test', download=True, transform=transform_ds8)

testloader = torch.utils.data.DataLoader(
    testset, batch_size=N_VALID, shuffle=False, num_workers=1)

probs = get_train_probs()
classes = probs[0].detach().numpy()
np.savetxt("svhn8_train_dlaactuals.csv",classes,delimiter=",")
probs = probs[1].detach().numpy()
np.savetxt("svhn8_train_dlaprobs.csv",probs,delimiter=",")

probs = get_test_probs()
classes = probs[0].detach().numpy()
np.savetxt("svhn8_test_dlaactuals.csv",classes,delimiter=",")
probs = probs[1].detach().numpy()
np.savetxt("svhn8_test_dlaprobs.csv",probs,delimiter=",")

trainset = torchvision.datasets.SVHN(
    root='./svhndata', split='train', download=True, transform=transform_ds16)

trainloader = torch.utils.data.DataLoader(
    trainset, batch_size=N_TRAIN, shuffle=False, num_workers=1)

testset = torchvision.datasets.SVHN(
    root='./svhndata', split='test', download=True, transform=transform_ds16)

testloader = torch.utils.data.DataLoader(
    testset, batch_size=N_VALID, shuffle=False, num_workers=1)

probs = get_train_probs()
classes = probs[0].detach().numpy()
np.savetxt("svhn16_train_dlaactuals.csv",classes,delimiter=",")
probs = probs[1].detach().numpy()
np.savetxt("svhn16_train_dlaprobs.csv",probs,delimiter=",")

probs = get_test_probs()
classes = probs[0].detach().numpy()
np.savetxt("svhn16_test_dlaactuals.csv",classes,delimiter=",")
probs = probs[1].detach().numpy()
np.savetxt("svhn16_test_dlaprobs.csv",probs,delimiter=",")


















