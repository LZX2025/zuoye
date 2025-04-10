# -*- coding: gbk -*-

import numpy as np
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import StepLR
import torch.optim as optim

filepath = 'C:/Users/32673/Desktop/工作室作业/小组/week3'


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channel, out_channel, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channel, out_channel, kernel_size=3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channel)
        self.conv2 = nn.Conv2d(out_channel, out_channel, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channel)

        self.relu = nn.ReLU(inplace=True)
        self.shortcut = nn.Sequential()
        self.stride = stride
        if stride != 1 or in_channel != out_channel * self.expansion :
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_channel, out_channel * self.expansion, kernel_size=1, stride=self.stride),
                nn.BatchNorm2d(out_channel * self.expansion))



    def forward(self, x):
        res = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += self.shortcut(res)
        out = self.relu(out)
        return out

class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super().__init__()
        self.in_channel = 64
        self.expansion = 1
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1)    #32*32不用池化吧[思考]
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)    # 我不造啊他们说用四层我就写了四个layer
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(self, block, out_channel, num_blocks, stride):
        layers = []
        layers.append(block(self.in_channel, out_channel, stride))
        self.in_channel = out_channel * self.expansion
        for _ in range(1,num_blocks):
            layers.append(block(self.in_channel, out_channel))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

def resnet(num_classes=10):
    model = ResNet(BasicBlock, [2, 2, 2, 2], num_classes=num_classes)    # 不造啊他们说两个block比较好我就这样写了
    return model

def get_data(batch_size=128):# 数据预处理
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

    # 加载数据集
    trainset = torchvision.datasets.CIFAR10(root=filepath, train=True, download=True, transform=transform_train)
    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2)

    testset = torchvision.datasets.CIFAR10(root=filepath, train=False, download=True, transform=transform_test)
    testloader = DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=2)

    return trainloader, testloader

##
def train(model, device, train_loader, optimizer, criterion, epoch):
    model.train()
    train_loss = 0
    correct = 0
    total = 0

    for batch_idx, (inputs, targets) in enumerate(train_loader):
        inputs, targets = inputs.to(device), targets.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)    # 好像有说规范要用'_'忽略掉用不到的' max value'
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        if batch_idx % 100 == 0:
            print(f'Epoch: {epoch} | Batch: {batch_idx}/{len(train_loader)} | Loss: {loss.item():.3f}')    # 不知道这个格式有没有什么规范所以网上超的

    acc = 100. * correct / total
    avg_loss = train_loss / len(train_loader)
    print(f'Train Epoch: {epoch} | Average Loss: {avg_loss:.4f} | Accuracy: {acc:.2f}%')    # 同上

    return avg_loss, acc

##
def test(model, device, test_loader, criterion):
    model.eval()
    test_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(test_loader):
            inputs, targets = inputs.to(device), targets.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

    acc = 100. * correct / total
    avg_loss = test_loss / len(test_loader)
    print(f'Test Loss: {avg_loss:.4f} | Accuracy: {acc:.2f}%')

    return avg_loss, acc

def load_saved_model(model_path, device):
    model = resnet().to(device)
    model.load_state_dict(torch.load(model_path))
    # 评估模式
    model.eval()
    return model


def main():
    # 超参数设置
    batch_size = 128
    epochs = 100
    learning_rate = 0.1
    momentum = 0.9    # 动量因子(原理暂时不清楚)
    weight_decay = 5e-4
    gamma = 0.1
    #milestones = [50, 75]

    # 设备配置
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_loader, test_loader = get_data(batch_size)

    model = resnet().to(device)
    # 各种优化, 交叉熵损失函数
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum, weight_decay=weight_decay)
    scheduler = StepLR(optimizer, step_size=30, gamma=gamma)

    # 训练和测试循环
    train_losses, train_accs = [], []
    test_losses, test_accs = [], []

    for epoch in range(1, epochs + 1):
        print(f"\nEpoch {epoch}/{epochs}")
        train_loss, train_acc = train(model, device, train_loader, optimizer, criterion, epoch)
        test_loss, test_acc = test(model, device, test_loader, criterion)

        train_losses.append(train_loss)
        train_accs.append(train_acc)
        test_losses.append(test_loss)
        test_accs.append(test_acc)

        scheduler.step()



    # 保存模型
    torch.save(model.state_dict(), filepath + '/resnet_model.pth')


if __name__ == '__main__':

    # LOAD_AND_TEST
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        print("Using GPU")
    else:
        print("Using CPU")
    loaded_model = load_saved_model(filepath + '/resnet_model.pth', device)
    _, test_loader = get_data(128)

    # 测试加载的模型
    criterion = nn.CrossEntropyLoss()
    test_loss, test_acc = test(loaded_model, device, test_loader, criterion)
    print(f"Loaded model test accuracy: {test_acc:.2f}%")

    # TRAIN_
    #main()
