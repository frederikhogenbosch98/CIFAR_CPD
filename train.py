import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms

import tqdm
import numpy as np
from models.resnet import ResNet


transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

trainset = torchvision.datasets.CIFAR10(root='/data', train=True, download=True, transform=transform_train)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=256, shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(root='/data', train=False, download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=128, shuffle=False, num_workers=2)

classes = ['plane', 'car', 'bird', 'cat', 'deer', \
           'dog', 'frog', 'horse', 'ship', 'truck']


if torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

NUM_EPCOHS = 100

model = ResNet(block_depths=[2,2,6,2])

optimizer = torch.optim.Adam(model.parameters(),
                                    lr=1e-4, 
                                    weight_decay=1e-4)
criterion = nn.CrossEntropyLoss()

scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=25, gamma=0.25) 

for epoch in range(NUM_EPCOHS):
    with tqdm.tqdm(trainloader, unit="batch", leave=False) as tepoch:
        for images, labels in tqdm.tqdm(tepoch):
            images, labels = images.to(device), labels.to(device)
            output = model(images)
            loss = criterion(output, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        scheduler.step()


model.to(device)
model.eval()


correct = 0
total = 0
test_accuracy = []
with torch.no_grad():
    for images, labels in testloader:
        images, labels = images.to(device), labels.to(device)
        output = model(images)
        _, predicted = torch.max(output.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    

accuracy = 100 * correct / total
print(f'Accuracy: {np.round(accuracy,3)}%')

