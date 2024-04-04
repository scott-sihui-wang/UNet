from model import UNet
from dataloader import Cell_data

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import torch.optim as optim
import matplotlib.pyplot as plt

import os

#import any other libraries you need below this line

import numpy as np
import datetime

# Paramteres

# learning rate
lr = 0.01
# number of training epochs
epoch_n = 10
# input image-mask size
image_size = 572
# root directory of project
root_dir = os.getcwd()
# training batch size
batch_size = 4
# use checkpoint model for training
load = True
# use GPU for training
gpu = False

data_dir = os.path.join(root_dir, 'data/cells')

trainset = Cell_data(data_dir=data_dir, size=image_size)
trainloader = DataLoader(trainset, batch_size=4, shuffle=True)

testset = Cell_data(data_dir=data_dir, size=image_size, train=False)
testloader = DataLoader(testset, batch_size=4)

device = torch.device('cuda:0' if gpu else 'cpu')

model = UNet(n_classes = 2).to('cuda:0').to(device)

if load:
    print('loading model')
    model.load_state_dict(torch.load('checkpoint.pt'))

criterion = nn.CrossEntropyLoss()

optimizer = optim.Adam(model.parameters(), lr=lr, betas=(0.9,0.99), weight_decay=0.0005)

model.train()
train_losses = []
test_losses = []

starttime=datetime.datetime.now()

for e in range(epoch_n):
    epoch_loss = 0
    model.train()
    for i, data in enumerate(trainloader):
        image, label = data

        image = image.to(device)
        label = label.long().squeeze(1).to(device)

        pred = model(image)

        crop_x = (label.shape[1] - pred.shape[2]) // 2
        crop_y = (label.shape[2] - pred.shape[3]) // 2

        label = label[:, crop_x: label.shape[1] - crop_x, crop_y: label.shape[2] - crop_y]

        loss = criterion(pred, label)

        loss.backward()

        optimizer.step()
        optimizer.zero_grad()

        epoch_loss += loss.item()

        print('batch %d --- Loss: %.4f' % (i, loss.item() / batch_size))
    print('Epoch %d / %d --- Loss: %.4f' % (e + 1, epoch_n, epoch_loss / trainset.__len__()))
    train_losses.append(epoch_loss / trainset.__len__())

    torch.save(model.state_dict(), 'checkpoint_epoch'+str(e+1)+'.pt')

    model.eval()

    total = 0
    correct = 0
    total_loss = 0

    with torch.no_grad():
        for i, data in enumerate(testloader):
            image, label = data

            image = image.to(device)
            label = label.long().squeeze(1).to(device)

            pred = model(image)
            crop_x = (label.shape[1] - pred.shape[2]) // 2
            crop_y = (label.shape[2] - pred.shape[3]) // 2

            label = label[:, crop_x: label.shape[1] - crop_x, crop_y: label.shape[2] - crop_y]

            loss = criterion(pred, label)
            total_loss += loss.item()

            _, pred_labels = torch.max(pred, dim=1)

            total += label.shape[0] * label.shape[1] * label.shape[2]
            correct += (pred_labels == label).sum().item()

        print('Accuracy: %.4f ---- Loss: %.4f' % (correct / total, total_loss / testset.__len__()))
        test_losses.append(total_loss / testset.__len__())

x_axis = np.arange(epoch_n) + 1
plt.plot(x_axis,train_losses,'b.-')
plt.plot(x_axis,test_losses,'r.-')
plt.ylabel('average error')
plt.legend(['Training loss','Validation loss'])
plt.xlabel('Epoch')
plt.show()

endtime=datetime.datetime.now()
print("Total Time:",endtime-starttime)

#testing and visualization

model.eval()

output_masks = []
output_labels = []

with torch.no_grad():
    for i in range(testset.__len__()):
        image, labels = testset.__getitem__(i)

        #input_image = image.unsqueeze(0).unsqueeze(0).to(device)
        input_image = image.unsqueeze(0).to(device)
        labels = labels.squeeze(0).squeeze(0).to(device)
        pred = model(input_image)

        output_mask = torch.max(pred, dim=1)[1].cpu().squeeze(0).numpy()
        output_mask = output_mask * 255.
        #print("output_mask:",output_mask)
        #print(np.max(output_mask))
        #print(np.min(output_mask))

        crop_x = (labels.shape[0] - output_mask.shape[0]) // 2
        crop_y = (labels.shape[1] - output_mask.shape[1]) // 2
        labels = labels[crop_x: labels.shape[0] - crop_x, crop_y: labels.shape[1] - crop_y].numpy()
        labels = labels * 255.
        #print("labels:",labels)
        #print(np.max(labels))
        #print(np.min(labels))

        output_masks.append(output_mask)
        output_labels.append(labels)

fig, axes = plt.subplots(testset.__len__(), 2, figsize = (20, 20))

for i in range(testset.__len__()):
  axes[i, 0].imshow(output_labels[i])
  axes[i, 0].axis('off')
  axes[i, 1].imshow(output_masks[i])
  axes[i, 1].axis('off')
plt.show()
