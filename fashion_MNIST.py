import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
from sklearn.metrics import confusion_matrix
 

#baseline accuracy = 90.19

train_transform = transforms.Compose([
    #transforms.RandomRotation(5),
    #transforms.RandomCrop(28, padding=2),
    transforms.ToTensor()
])

val_transform = transforms.Compose([
    transforms.ToTensor()
])



test_transform = transforms.Compose([
    transforms.ToTensor()
])

train_data = datasets.FashionMNIST(
    root = "data",
    train = True,
    download = True,
    transform=train_transform
)

train_loader = DataLoader(train_data, batch_size = 64)

val_data = datasets.FashionMNIST(
    root = "data",
    train = True,
    download = True,
    transform=val_transform
) 

val_loader = DataLoader(val_data, batch_size = 64)



test_data = datasets.FashionMNIST(
    root = "data",
    train = False,
    download = True,
    transform=test_transform
)


test_loader = DataLoader(test_data, batch_size = 64)


print(len(train_data))
print(len(test_data))

train_data, val_data = random_split(train_data, [48000, 12000])  


class FashionCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels = 1,
                               out_channels =8,
                               kernel_size = 3)
        self.bn1 = nn.BatchNorm2d(8)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(kernel_size = 2)
        self.conv2 = nn.Conv2d(in_channels =8,
                               out_channels = 13,
                               kernel_size = 3
        )
        self.bn2 = nn.BatchNorm2d(13)
        self.relu = nn.ReLU() 
        self.dropout = nn.Dropout(p=0.3)
        self.final_layer = nn.Linear(1573,10)

    def forward(self,x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.pool(x)
        #x = x.view(x.size(0), -1)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = torch.flatten(x, start_dim = 1)
        x = self.final_layer(x)
        return x
        

model = FashionCNN()
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.001
)

train_losses = []
val_losses = []

train_loss_sum = 0
val_loss_sum = 0

correct = 0 
incorrect = 0

for i in range(10):
    model.train()
    for x, y in train_loader:
        prediction = model(x)
        loss = criterion(prediction, y)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        train_loss_sum+=loss.item()
        print("train loss: ", loss.item())


    avg_train_loss = train_loss_sum/len(train_loader)
    train_losses.append(train_loss_sum)

    max_val_loss = float("inf")

    model.eval()
    with torch.no_grad():
        for x_val,y_val in val_loader:
            val_prediction = model(x_val)
            val_loss = criterion(val_prediction, y_val)
            val_loss_sum+=loss.item()
            print("val loss: ", loss.item())

    avg_val_loss = val_loss_sum/len(val_loader)
    val_losses.append(avg_val_loss)
    
    if avg_val_loss < max_val_loss:
        max_val_loss = avg_val_loss
        torch.save(model.state_dict(), "best_mnist_model.pth")
        



correct = 0 
total = 0

all_pred = []
all_labs = []

val_losses.append(val_loss_sum)
with torch.no_grad():
    for x,y in test_loader:
        test_prediction = model(x)
        predicted_classes = torch.argmax(test_prediction, dim=1)
        all_pred.extend(predicted_classes.tolist())
        all_labs.extend(y.tolist()) 
        correct+= (predicted_classes == y).sum().item()
        total+= y.size(0)

test_acc = correct/total
print(test_acc)
        



    



