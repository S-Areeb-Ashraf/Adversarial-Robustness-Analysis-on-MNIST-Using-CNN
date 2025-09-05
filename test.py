import torch
import time
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import pandas as pd
import numpy as np
from fgsm import generate_adversarial_example
from fgsm_gaussian import generate_gaussian_adversarial_example

t_accuracy=0.0

class SimpleNet(nn.Module):
    def __init__(self):
        super(SimpleNet, self).__init__()
        self.conv1=nn.Conv2d(1,32,3,1)
        self.conv2=nn.Conv2d(32,64,3,1)
        self.dropout1=nn.Dropout(0.25)
        self.dropout2=nn.Dropout(0.5)
        self.linear1=nn.Linear(64*12*12,128)
        self.linear2=nn.Linear(128,10)
        self.sofmax=nn.Softmax(dim=1)

    def forward(self,x):
        x=self.conv1(x)
        x=F.relu(x)
        # x=self.(x)
        x=self.conv2(x)
        x=F.relu(x)
        x=F.max_pool2d(x,2)
        x=self.dropout1(x)
        x=torch.flatten(x,1)
        x=self.linear1(x)
        x=F.relu(x)
        x=self.dropout2(x)
        x=self.linear2(x)
        output=self.sofmax(x)
        return output


def train_model(num_epochs,dataloader,optimizer,model,criteriaon):
    model.train()
    for epoch in range(num_epochs):
        train_loss=0.0
        correct=0
        for data in dataloader:
            optimizer.zero_grad()
            features,labels=data
            output=model(features)
            loss=criteriaon(output,labels)
            loss.backward()
            optimizer.step()
            train_loss+=loss.item()* features.size(0)
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(labels.view_as(pred)).sum().item()

        epoch_loss=train_loss/len(dataloader.dataset)
        accuracy = 100. * correct / len(dataloader.dataset)
        print(f"Epoch {epoch+1}, Training Loss: {epoch_loss:.4f}, Accuracy: {correct}/{len(train_loader.dataset)} ({accuracy:.2f}%)'")


def test_model(dataloader,model):
    v_loss=0.0
    correct=0
    model.eval()
    with torch.no_grad():
        for data in dataloader:
            features,labels=data
            output=model(features)
            loss=criteriaon(output,labels)
            v_loss+=loss.item()* features.size(0)
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(labels.view_as(pred)).sum().item()

        v_loss/=len(dataloader.dataset)
        accuracy = 100. * correct / len(dataloader.dataset)
        global t_accuracy
        t_accuracy=accuracy
        print(f"Validation Loss: {v_loss:.4f}, Accuracy: {correct}/{len(dataloader.dataset)} ({accuracy:.2f}%)")


def evaluate_robustness(model, test_loader, epsilon_values, method='fgsm'):
    for epsilon in epsilon_values:
        correct = 0
        total = 0

        for data, target in test_loader:
            for i in range(len(data)):
                original, adversarial, pred = generate_adversarial_example(
                    model, data[i].unsqueeze(0), target[i].unsqueeze(0), epsilon
                )

                if pred.item() == target[i].item():
                    correct += 1
                total += 1


        accuracy = 100. * correct / total
        accuracy_drop = t_accuracy - accuracy
        print(f"Epsilon: {epsilon:.3f}\t\t Accuracy {accuracy:.2f}%\t\t Accuracy Drop: {accuracy_drop:.2f}%")

def evaluate_gaussian(model, test_loader, epsilon_values):
    for epsilon in epsilon_values:
        correct = 0
        total = 0

        for data, target in test_loader:
            for i in range(len(data)):
                original, adversarial, pred = generate_gaussian_adversarial_example(
                    model, data[i].unsqueeze(0), target[i].unsqueeze(0), epsilon
                )
                if pred.item() == target[i].item():
                    correct += 1
                total += 1
        accuracy = 100. * correct / total
        accuracy_drop = t_accuracy - accuracy  
        print(f"Epsilon: {epsilon:.3f}\t\t Accuracy {accuracy:.2f}%\t\t Accuracy Drop: {accuracy_drop:.2f}%")

transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
train_dataset = datasets.MNIST('data', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST('data', train=False, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

model = SimpleNet()

optimizer = optim.Adam(model.parameters(), lr=0.001)
criteriaon=nn.CrossEntropyLoss()
train_model(5,train_loader,optimizer,model,criteriaon)
test_model(test_loader,model)
model.train()
epsilon_values = [0.01,0.05,0.1,0.25,0.2,0.3,0.5]
evaluate_robustness(model, test_loader, epsilon_values, method='fgsm')
time.sleep(5)
evaluate_gaussian(model, test_loader, epsilon_values)
