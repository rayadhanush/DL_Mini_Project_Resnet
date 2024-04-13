# %%
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms

import numpy as np
import seaborn as sn
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import confusion_matrix, classification_report

import os
import argparse

# %%
!pip install torchviz
!pip install graphviz


# %%
class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=1, stride=stride, padding=0, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=1,
                               stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion *
                               planes, kernel_size=3, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion*planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNet, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512*block.expansion, num_classes)
        self.metric = 0

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


def ResNet18():
    return ResNet(BasicBlock, [1, 2, 2, 1])


def ResNet34():
    return ResNet(BasicBlock, [3, 4, 6, 3])


def ResNet50():
    return ResNet(Bottleneck, [3, 4, 6, 3])


def ResNet101():
    return ResNet(Bottleneck, [3, 4, 23, 3])


def ResNet152():
    return ResNet(Bottleneck, [3, 8, 36, 3])


# def test():
#     net = ResNet18()
#     y = net(torch.randn(1, 3, 32, 32))
#     print(y.size())


# %%
device = 'cuda' if torch.cuda.is_available() else 'cpu'
best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch

# %%
net = ResNet18()
net = net.to(device)
if device == 'cuda':
  net = torch.nn.DataParallel(net)
  cudnn.benchmark = True

# %%
total_params = sum(p.numel() for p in net.parameters() if p.requires_grad)
print(total_params)

# %%
#Data Augmentation
class CutOut(object):
    def __init__(self, n_holes, length):
        self.n_holes = n_holes
        self.length = length

    def __call__(self, img):
        """
        img: Tensor image of size (C, H, W).
        """
        h = img.size(1)
        w = img.size(2)
        mask = np.ones((h, w), np.float32)

        for n in range(self.n_holes):
            y = np.random.randint(h)
            x = np.random.randint(w)

            y1 = np.clip(y - self.length // 2, 0, h)
            y2 = np.clip(y + self.length // 2, 0, h)
            x1 = np.clip(x - self.length // 2, 0, w)
            x2 = np.clip(x + self.length // 2, 0, w)

            mask[y1: y2, x1: x2] = 0.

        mask = torch.from_numpy(mask)
        mask = mask.expand_as(img)
        img *= mask

        return img


# %%
import pickle
import os
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    CutOut(n_holes=1, length=16),  
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

#Unpickle CIFAR Data
def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

# Custom Dataset class to load the train and val datasets
class CIFAR10Dataset(Dataset):
    def __init__(self, file, transform=None):
        data_dict = unpickle(file)
        self.data = data_dict[b'data']
        self.labels = data_dict[b'labels']
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image = self.data[idx].reshape(3, 32, 32).transpose(1, 2, 0)
        label = self.labels[idx]

        image = Image.fromarray(image)

        if self.transform:
            image = self.transform(image)

        return image, label

# Paths to the CIFAR-10 dataset files
data_path = '/scratch/aup1007/project/dl_data/cifar-10-python/cifar-10-batches-py'  
train_files = [os.path.join(data_path, 'data_batch_{}'.format(i)) for i in range(1, 6)]
val_files = os.path.join(data_path, 'test_batch')

# Training dataset and loader
trainset = torch.utils.data.ConcatDataset([CIFAR10Dataset(batch_file, transform=transform_train) for batch_file in train_files])
trainloader = DataLoader(trainset, batch_size=32, shuffle=True, num_workers=2)

# Val dataset and loader
valset = CIFAR10Dataset(val_files, transform=transform_test)
valloader = DataLoader(valset, shuffle=False, num_workers=2)

# Loading label names
meta_data = unpickle(os.path.join(data_path, 'batches.meta'))
classes = meta_data[b'label_names']
classes = [label.decode('utf-8') for label in classes]


# %%
# We need to convert the images to numpy arrays as tensors are not compatible with matplotlib.
def im_convert(tensor):  
  image = tensor.cpu().clone().detach().numpy() 
  image = image.transpose(1, 2, 0)
  image = image * np.array((0.5, 0.5, 0.5)) + np.array((0.5, 0.5, 0.5))
  image = image.clip(0, 1)
  return image

data_iterable = iter(trainloader) # converting our train_dataloader to iterable so that we can iter through it. 
images, labels = next(data_iterable) #going from 1st batch of 100 images to the next batch
fig = plt.figure(figsize=(20, 10)) 

for idx in np.arange(10):
    ax = fig.add_subplot(2, 5, idx+1, xticks=[], yticks=[])
    plt.imshow(im_convert(images[idx]))
    ax.set_title(classes[labels[idx].item()])

# %%
from torch.optim import optimizer


def train(epoch):
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    train_losses = []
    train_acc = []
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
        train_losses.append(train_loss)
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
        train_acc.append(100.*correct/total)


    acc = np.array(train_acc).sum()/len(trainloader)
    loss = np.array(train_loss).sum()/len(trainloader)
    print("\nTrain Loss:",loss,"Train acc:",acc)
    return loss,acc

# %%
def val(epoch):
    global best_acc
    net.eval()
    val_loss = 0
    val_losses = []
    val_acc = []
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(valloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            val_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            
            val_acc.append(100.*correct/total)
            val_losses.append(val_loss)
        loss = np.array(val_loss).sum()/len(valloader)
        acc = np.array(val_acc).sum()/len(valloader)
        print("\nValidation loss",loss,"Validation Acc:",acc)
    return loss,acc

# %%
optimizerDict = {
                    "SGD" : optim.SGD(net.parameters(), lr=0.1,
                        momentum=0.9, weight_decay=5e-4),
                    "Adam" : optim.Adam(net.parameters(),lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=5e-4, 
                        amsgrad=False, foreach=None, maximize=False, capturable=False),
                    "Adadelta" : optim.Adadelta(net.parameters(), lr=0.1, rho=0.9, eps=1e-08, 
                        weight_decay=5e-4, foreach=None, maximize=False),
                    "Adagrad" : optim.Adagrad(net.parameters(), lr=0.01, lr_decay=0, 
                        weight_decay=5e-4, initial_accumulator_value=0, eps=1e-08, foreach=None, maximize=False)
                }

# %%
import time

train_loss_history_Adagrad = []
train_acc_history_Adagrad = []
val_loss_history_Adagrad = []
val_acc_history_Adagrad = []

epoch = 0

print('==> Building model..')

early_stopping_patience = 3  # Number of epochs to wait after min has been hit
consecutive_epochs_meeting_criterion = 0  # Counter for epochs meeting the criterion
target_loss = 0.3

optimizer = optimizerDict["Adagrad"]
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min')
print("Using optimizer:","Adagrad")
criterion = nn.CrossEntropyLoss()
while epoch<100:
    start_time = time.time()
    epoch+=1
    train_loss,train_acc = train(epoch)
    val_loss,val_acc = val(epoch)
    train_loss_history_Adagrad.append(train_loss)
    train_acc_history_Adagrad.append(train_acc)
    val_loss_history_Adagrad.append(val_loss)
    val_acc_history_Adagrad.append(val_acc)
    
    if val_loss <= target_loss:
        consecutive_epochs_meeting_criterion += 1
        if consecutive_epochs_meeting_criterion >= early_stopping_patience:
            print(f"Early stopping triggered at epoch {epoch}: Validation loss reached {val_loss}% for {consecutive_epochs_meeting_criterion} consecutive epochs.")
            break  # Stop training if condition is met
    else:
        consecutive_epochs_meeting_criterion = 0
    
    
    scheduler.step(val_loss)
    end_time = time.time()  
    elapsed_time = end_time - start_time  
    print(f'Epoch {epoch} completed in {elapsed_time:.2f} seconds.')
print(best_acc)


# %%
plt.plot(train_loss_history_Adagrad, 'r', label='Training Loss')
plt.plot(val_loss_history_Adagrad,  'g',  label='Validation Loss')
plt.xlabel("Epoch")
plt.title("Loss (Model with Adagrad Optimizer)")
plt.legend()

# %%
plt.plot(train_acc_history_Adagrad, 'r', label='Training Accuracy')
plt.plot(val_acc_history_Adagrad,  'g',  label='Validation Accuracy')
plt.xlabel("Epoch")
plt.title("Accuracy (Model with Adagrad Optimizer)")
plt.legend()

# %%
y_pred = []
y_true = []

# iterate over test data
with torch.no_grad():
    for batch_idx, (inputs, labels) in enumerate(testloader):
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = net(inputs)

        _, pred = outputs.max(1)
        y_pred.extend(pred.data.cpu().numpy())
        y_true.extend(labels.data.cpu().numpy())

# Build confusion matrix
cf_matrix = confusion_matrix(y_true, y_pred)
df_cm = pd.DataFrame(cf_matrix/np.sum(cf_matrix) *10, index = [i for i in classes],
                     columns = [i for i in classes])
plt.figure(figsize = (12,7))
sn.heatmap(df_cm, annot=True)
plt.savefig('ConfusionMatrix(Adagrad).png')

# %%

total_params = sum(p.numel() for p in net.parameters() if p.requires_grad)
print(total_params)

# %%
acc_array_Adagrad

# %%
if 'net' in locals():
    del net  
    torch.cuda.empty_cache()  
net = ResNet18().to(device)
if device == 'cuda':
    net = torch.nn.DataParallel(net)
    cudnn.benchmark = True


# %%
import time

train_loss_history_Adadelta = []
train_acc_history_Adadelta = []
val_loss_history_Adadelta = []
val_acc_history_Adadelta = []
epoch = 0
print('==> Building model..')

early_stopping_patience = 3  # Number of epochs to wait after min has been hit
consecutive_epochs_meeting_criterion = 0  # Counter for epochs meeting the criterion
target_loss = 0.3 

optimizer = optimizerDict["Adadelta"]
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min')
print("Using optimizer:","Adadelta")
criterion = nn.CrossEntropyLoss()
while epoch<100:
    start_time = time.time()
    epoch+=1
    train_loss,train_acc = train(epoch)
    val_loss,val_acc = val(epoch)
    train_loss_history_Adadelta.append(train_loss)
    train_acc_history_Adadelta.append(train_acc)
    val_loss_history_Adadelta.append(val_loss)
    val_acc_history_Adadelta.append(val_acc)
    
    if val_loss <= target_loss:
        consecutive_epochs_meeting_criterion += 1
        if consecutive_epochs_meeting_criterion >= early_stopping_patience:
            print(f"Early stopping triggered at epoch {epoch}: Validation loss reached {val_loss}% for {consecutive_epochs_meeting_criterion} consecutive epochs.")
            break  # Stop training if condition is met
    else:
        consecutive_epochs_meeting_criterion = 0
    
    
    scheduler.step(val_loss)
    end_time = time.time()  
    elapsed_time = end_time - start_time  
    print(f'Epoch {epoch} completed in {elapsed_time:.2f} seconds.')
print(best_acc)


# %%
plt.plot(train_loss_history_Adadelta, 'r', label='Training Loss')
plt.plot(val_loss_history_Adadelta,  'g',  label='Validation Loss')
plt.xlabel("Epoch")
plt.title("Loss (Model with Adadelta Optimizer)")
plt.legend()

# %%
plt.plot(train_acc_history_Adadelta, 'r', label='Training Accuracy')
plt.plot(val_acc_history_Adadelta,  'g',  label='Validation Accuracy')
plt.xlabel("Epoch")
plt.title("Accuracy (Model with Adadelta Optimizer)")
plt.legend()

# %%
y_pred = []
y_true = []

# iterate over test data
with torch.no_grad():
    for batch_idx, (inputs, labels) in enumerate(testloader):
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = net(inputs)

        _, pred = outputs.max(1)
        y_pred.extend(pred.data.cpu().numpy())
        y_true.extend(labels.data.cpu().numpy())

# Build confusion matrix
cf_matrix = confusion_matrix(y_true, y_pred)
df_cm = pd.DataFrame(cf_matrix/np.sum(cf_matrix) *10, index = [i for i in classes],
                     columns = [i for i in classes])
plt.figure(figsize = (12,7))
sn.heatmap(df_cm, annot=True)
plt.savefig('ConfusionMatrix(Adadelta).png')

# %%
if 'net' in locals():
    del net  
    torch.cuda.empty_cache()  
net = ResNet18().to(device)
if device == 'cuda':
    net = torch.nn.DataParallel(net)
    cudnn.benchmark = True


# %%
import time


train_loss_history_Adam = []
train_acc_history_Adam = []
val_loss_history_Adam = []
val_acc_history_Adam = []


epoch = 0
print('==> Building model..')

early_stopping_patience = 3  # Number of epochs to wait after min has been hit
consecutive_epochs_meeting_criterion = 0  # Counter for epochs meeting the criterion
target_loss = 0.3

optimizer = optimizerDict["Adam"]
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min')
print("Using optimizer:","Adam")
criterion = nn.CrossEntropyLoss()
while epoch<100:
    start_time = time.time()
    epoch+=1
    train_loss,train_acc = train(epoch)
    val_loss,val_acc = test(epoch)
    train_loss_history_Adam.append(train_loss)
    train_acc_history_Adam.append(train_acc)
    val_loss_history_Adam.append(val_loss)
    val_acc_history_Adam.append(val_acc)
    
    if val_loss <= target_loss:
        consecutive_epochs_meeting_criterion += 1
        if consecutive_epochs_meeting_criterion >= early_stopping_patience:
            print(f"Early stopping triggered at epoch {epoch}: Validation loss reached {val_loss}% for {consecutive_epochs_meeting_criterion} consecutive epochs.")
            break  # Stop training if condition is met
    else:
        consecutive_epochs_meeting_criterion = 0
    
    
    scheduler.step(val_loss)
    end_time = time.time()  
    elapsed_time = end_time - start_time  
    print(f'Epoch {epoch} completed in {elapsed_time:.2f} seconds.')
print(best_acc)


# %%
data = {
    "Train loss adam": train_loss_history_Adam,
"train acc adam": train_acc_history_Adam,
"val loss adam":  val_loss_history_Adam,
"val acc adam": val_acc_history_Adam
}

pd.DataFrame(data).to_csv('adam.csv', index=False)

# %%
plt.plot(train_loss_history_Adam, 'r', label='Training Loss')
plt.plot(val_loss_history_Adam,  'g',  label='Validation Loss')
plt.xlabel("Epoch")
plt.title("Loss (Model with Adam Optimizer)")
plt.legend()

# %%
plt.plot(train_acc_history_Adam, 'r', label='Training Accuracy')
plt.plot(val_acc_history_Adam,  'g',  label='Validation Accuracy')
plt.xlabel("Epoch")
plt.title("Accuracy (Model with Adam Optimizer)")
plt.legend()

# %%
y_pred = []
y_true = []

# iterate over test data
with torch.no_grad():
    for batch_idx, (inputs, labels) in enumerate(testloader):
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = net(inputs)

        _, pred = outputs.max(1)
        y_pred.extend(pred.data.cpu().numpy())
        y_true.extend(labels.data.cpu().numpy())

# Build confusion matrix
cf_matrix = confusion_matrix(y_true, y_pred)
df_cm = pd.DataFrame(cf_matrix/np.sum(cf_matrix) *10, index = [i for i in classes],
                     columns = [i for i in classes])
plt.figure(figsize = (12,7))
sn.heatmap(df_cm, annot=True)
plt.savefig('ConfusionMatrix(Adam).png')

# %%
if 'net' in locals():
    del net  
    torch.cuda.empty_cache()  
net = ResNet18().to(device)
if device == 'cuda':
    net = torch.nn.DataParallel(net)
    cudnn.benchmark = True


# %%
import time

train_loss_history_SGDN = []
train_acc_history_SGDN = []
val_loss_history_SGDN = []
val_acc_history_SGDN = []


epoch = 0
print('==> Building model..')

early_stopping_patience = 3  # Number of epochs to wait after min has been hit
consecutive_epochs_meeting_criterion = 0  # Counter for epochs meeting the criterion
target_loss = 0.3 

optimizer = optimizerDict["SGDN"]
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min')
print("Using optimizer:","SGDN")
criterion = nn.CrossEntropyLoss()
while epoch<100:
    start_time = time.time()
    epoch+=1
    train_loss,train_acc = train(epoch)
    val_loss,val_acc = test(epoch)
    train_loss_history_SGDN.append(train_loss)
    train_acc_history_SGDN.append(train_acc)
    val_loss_history_SGDN.append(val_loss)
    val_acc_history_SGDN.append(val_acc)
    
    if val_acc >= target_accuracy:
        consecutive_epochs_meeting_criterion += 1
        if consecutive_epochs_meeting_criterion >= early_stopping_patience:
            print(f"Early stopping triggered at epoch {epoch}: Validation loss reached {val_loss}% for {consecutive_epochs_meeting_criterion} consecutive epochs.")
            break  # Stop training if condition is met
    else:
        consecutive_epochs_meeting_criterion = 0
    
    
    scheduler.step(val_loss)
    end_time = time.time()  
    elapsed_time = end_time - start_time  
    print(f'Epoch {epoch} completed in {elapsed_time:.2f} seconds.')
print(best_acc)

# %%
plt.plot(train_loss_history_SGDN, 'r', label='Training Loss')
plt.plot(val_loss_history_SGDN,  'g',  label='Validation Loss')
plt.xlabel("Epoch")
plt.title("Loss (Model with SGDN Optimizer)")
plt.legend()

# %%
plt.plot(train_acc_history_SGDN, 'r', label='Training Accuracy')
plt.plot(val_acc_history_SGDN,  'g',  label='Validation Accuracy')
plt.xlabel("Epoch")
plt.title("Accuracy (Model with SGDN Optimizer)")
plt.legend()

# %%
y_pred = []
y_true = []

# iterate over test data
with torch.no_grad():
    for batch_idx, (inputs, labels) in enumerate(testloader):
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = net(inputs)

        _, pred = outputs.max(1)
        y_pred.extend(pred.data.cpu().numpy())
        y_true.extend(labels.data.cpu().numpy())

# Build confusion matrix
cf_matrix = confusion_matrix(y_true, y_pred)
df_cm = pd.DataFrame(cf_matrix/np.sum(cf_matrix) *10, index = [i for i in classes],
                     columns = [i for i in classes])
plt.figure(figsize = (12,7))
sn.heatmap(df_cm, annot=True)
plt.savefig('ConfusionMatrix(SGDN).png')

# %%
if 'net' in locals():
    del net  
    torch.cuda.empty_cache()  
net = ResNet18().to(device)
if device == 'cuda':
    net = torch.nn.DataParallel(net)
    cudnn.benchmark = True


# %%
import time

train_loss_history_SGD = []
train_acc_history_SGD = []
val_loss_history_SGD = []
val_acc_history_SGD = []


epoch = 0
print('==> Building model..')

early_stopping_patience = 3  # Number of epochs to wait after min has been hit
consecutive_epochs_meeting_criterion = 0  # Counter for epochs meeting the criterion
target_loss = 0.3 

optimizer = optimizerDict["SGD"]
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min')
print("Using optimizer:","SGD")
criterion = nn.CrossEntropyLoss()
while epoch<100:
    start_time = time.time()
    epoch+=1
    train_loss,train_acc = train(epoch)
    val_loss,val_acc = test(epoch)

    train_loss_history_SGD.append(train_loss)
    train_acc_history_SGD.append(train_acc)
    val_loss_history_SGD.append(val_loss)
    val_acc_history_SGD.append(val_acc)
    
    if val_acc >= target_accuracy:
        consecutive_epochs_meeting_criterion += 1
        if consecutive_epochs_meeting_criterion >= early_stopping_patience:
            print(f"Early stopping triggered at epoch {epoch}: Validation loss reached {val_loss}% for {consecutive_epochs_meeting_criterion} consecutive epochs.")
            break  # Stop training if condition is met
    else:
        consecutive_epochs_meeting_criterion = 0
    
    
    scheduler.step(val_loss)
    end_time = time.time()  
    elapsed_time = end_time - start_time  
    print(f'Epoch {epoch} completed in {elapsed_time:.2f} seconds.')
print(best_acc)


# %%
plt.plot(train_loss_history_SGD, 'r', label='Training Loss')
plt.plot(val_loss_history_SGD,  'g',  label='Validation Loss')
plt.xlabel("Epoch")
plt.title("Loss (Model with SGD Optimizer)")
plt.legend()

# %%
plt.plot(train_acc_history_SGD, 'r', label='Training Accuracy')
plt.plot(val_acc_history_SGD,  'g',  label='Validation Accuracy')
plt.xlabel("Epoch")
plt.title("Accuracy (Model with SGD Optimizer)")
plt.legend()

# %%
y_pred = []
y_true = []

# iterate over test data
with torch.no_grad():
    for batch_idx, (inputs, labels) in enumerate(testloader):
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = net(inputs)

        _, pred = outputs.max(1)
        y_pred.extend(pred.data.cpu().numpy())
        y_true.extend(labels.data.cpu().numpy())

# Build confusion matrix
cf_matrix = confusion_matrix(y_true, y_pred)
df_cm = pd.DataFrame(cf_matrix/np.sum(cf_matrix) *10, index = [i for i in classes],
                     columns = [i for i in classes])
plt.figure(figsize = (12,7))
sn.heatmap(df_cm, annot=True)
plt.savefig('ConfusionMatrix(SGD).png')

# %%
import matplotlib.pyplot as plt

def plot_optimizer_accuracy(accuracy, label):
    epochs = range(1, len(accuracy) + 1)
    plt.plot(epochs, accuracy, marker='o', linestyle='-', label=label)
    
plt.figure(figsize=(12, 8))

# Plotting the data for each optimizer
plot_optimizer_accuracy(val_acc_history_Adam, 'Adam Optimizer')
plot_optimizer_accuracy(val_acc_history_Adadelta, 'Adadelta Optimizer')
plot_optimizer_accuracy(val_acc_history_SGDN, 'SGDN Optimizer')
plot_optimizer_accuracy(val_acc_history_SGD, 'SGD Optimizer')
plot_optimizer_accuracy(val_acc_history_Adagrad, 'Adagrad Optimizer')

plt.title('Validation Accuracy vs. Number of Epochs')
plt.xlabel('Epochs')
plt.ylabel('Validation Accuracy')
plt.legend()
plt.grid(True)
plt.show()
plt.savefig('validation_accuracy_vs_epochs.png', dpi=300)


# %%
class CustomTestDataset(Dataset):
    def __init__(self, file, transform=None):
        data_dict = unpickle(file)
        self.data = data_dict[b'data']
        self.ids = data_dict[b'ids']
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image = self.data[idx].reshape(3, 32, 32).transpose(1, 2, 0)
        id = self.ids[idx]

        image = Image.fromarray(image)

        if self.transform:
            image = self.transform(image)

        return image, id

# %%
testset = CustomeTestDataset('./data/cifar_test_nolabels.pkl', transform=transform_test)
testloader = DataLoader(testset, shuffle=False, num_workers=2)

# %%
import csv

net.eval()
with open('test_pred.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['ID', 'Labels']) 

    with torch.no_grad():  
        for i, (inputs, ids) in enumerate(testloader):
            inputs = inputs.to(device)
            outputs = net(inputs)
            _, predicted = torch.max(outputs, 1)

            for id, pred in zip(ids, predicted):
                writer.writerow([id.item(), pred.item()])  

print("CSV file created: 'test_pred.csv'")


