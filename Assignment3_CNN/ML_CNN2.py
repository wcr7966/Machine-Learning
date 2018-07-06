# -*- coding: utf-8 -*-
import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=256,
                                          shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=256,
                                         shuffle=False, num_workers=2)

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 96, 3, 1, 1)
        self.conv2 = nn.Conv2d(96, 96, 3, 1, 1)
        self.conv3 = nn.Conv2d(96, 96, 3, 2, 1)     # pooling
        self.conv4 = nn.Conv2d(96, 192, 3, 1, 1)
        self.conv5 = nn.Conv2d(192, 192, 3, 1, 1)
        self.conv6 = nn.Conv2d(192, 192, 3, 2, 1)   # pooling
        self.conv7 = nn.Conv2d(192, 192, 3, 1, 0)
        self.conv8 = nn.Conv2d(192, 192, 1, 1, 0)
        self.conv9 = nn.Conv2d(192, 10, 1, 1, 0)
        self.pool = nn.AvgPool2d(6)

    def forward(self, x):
        d1 = F.dropout(x, 0.2, training=self.training)
        c1 = self.conv1(d1)
        r1 = F.relu(c1)
        c2 = self.conv2(r1)
        r2 = F.relu(c2)
        c3 = self.conv3(r2)
        r3 = F.relu(c3)
        d2 = F.dropout(r3, 0.2, training=self.training)
        c4 = self.conv4(d2)
        r4 = F.relu(c4)
        c5 = self.conv5(r4)
        r5 = F.relu(c5)
        c6 = self.conv6(r5)
        r6 = F.relu(c6)
        d3 = F.dropout(r6, 0.2, training=self.training)
        c7 = self.conv7(d3)
        r7 = F.relu(c7)
        c8 = self.conv8(r7)
        r8 = F.relu(c8)
        c9 = self.conv9(r8)
        r9 = F.relu(c9)
        x = self.pool(r9)
        x = x.view(-1, self.num_flat_features(x))
        return x


    def num_flat_features(self, x):
        size = x.size()[1:]
        num_features = 1
        for s in size:
            num_features *= s
        return num_features


net = Net()
if torch.cuda.is_available():
    net = net.cuda()
    print('using cuda')
else:
    print('using cpu')


import torch.optim as optim

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters(), lr=0.001, betas=(0.9, 0.99))


for epoch in range(150):  # loop over the dataset multiple times

    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        # get the inputs
        inputs, labels = data
        inputs, labels = Variable(inputs), Variable(labels)
        if torch.cuda.is_available():
            inputs, labels = inputs.cuda(), labels.cuda()

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.data[0]
        if i % 196 == 195:  # print every 2000 mini-batches
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / 196))
            running_loss = 0.0

print('Finished Training')


correct = 0
total = 0
for data in testloader:
    images, labels = data
    images, labels = Variable(images), Variable(labels)
    if torch.cuda.is_available():
        images, labels = images.cuda(), labels.cuda()
    outputs = net(images)
    _, predicted = torch.max(outputs, 1)
    total += labels.size(0)
    correct += float((predicted == labels).sum())

print('Accuracy of the network on the 10000 test images: %d %%' % (
        100 * correct / total))


class_correct = list(0. for i in range(10))
class_total = list(0. for i in range(10))
for data in testloader:
    images, labels = data
    images, labels = Variable(images), Variable(labels)
    if torch.cuda.is_available():
        images, labels = images.cuda(), labels.cuda()
    outputs = net(images)
    _, predicted = torch.max(outputs, 1)
    c = (predicted == labels).squeeze()
    for i in range(4):
        label = labels[i]
        class_correct[int(label.data)] += float(c[i].data[0])
        class_total[int(label.data)] += 1

for i in range(10):
    print('Accuracy of %5s : %2d %%' % (
        classes[i], 100 * class_correct[i] / class_total[i]))


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


print(device)


