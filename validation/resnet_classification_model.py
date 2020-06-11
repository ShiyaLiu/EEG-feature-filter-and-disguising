'''
ResNet-based classification model

modified the code from https://github.com/kuangliu/pytorch-cifar/blob/master/main.py
'''
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms

import os
import argparse

from resnet import ResNet18
from resnet import ResNet34
from resnet import ResNet50
from resnet import ResNet101

#from utils import progress_bar

import numpy as np
#from SyncNetModel import SyncNetModel
import scipy.io as sio
from torch.autograd import Variable
import torch.utils.data as Data
#import utils

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from sklearn.model_selection import train_test_split
from sklearn.metrics import recall_score
import pandas as pd
from sklearn import preprocessing


parser = argparse.ArgumentParser(description='PyTorch EEG images Training')
parser.add_argument('--model', type=str, default='ResNet18', help='which model')
parser.add_argument('--feature', required=True, type=str, help='select feature to be trained: alcoholism|stimulus|id')
parser.add_argument('--batchSize', type=int, default=64, help='input batch size')
parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')

args = parser.parse_args()

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Loading Data
print('==> Preparing data..')


class DataFrameDataset(torch.utils.data.Dataset):
    def __init__(self, data_input, label_input):
        self.data_tensor = torch.Tensor(data_input)
        self.label_tensor = torch.Tensor(label_input)
    # a function to get items by index
    def __getitem__(self, index):
        #obj = self.data_tensor[index]
        signal = self.data_tensor[index]
        target = self.label_tensor[index]

        return signal, target

    # a function to count samples
    def __len__(self):
        n = np.shape(self.data_tensor)[0]
        #print (n)
        return n
        n = np.shape(self.data_tensor)[0]
        return n
  
if not args.feature == 'id':  #load joint training set
    data_train = sio.loadmat('../datasets/eeg/eeg_images_train_augmented_within.mat')
else: #if classify id, load training set
    data_train = sio.loadmat('../datasets/eeg/uci_eeg_images_train_within.mat')
    
data_test = sio.loadmat('../datasets/eeg/uci_eeg_images_test_within.mat')

#data
X_train = np.transpose(data_train['data'],(0,3,2,1)).astype('float32')
X_test = np.transpose(data_test['data'],(0,3,2,1)).astype('float32')
#label
label = 'label_%s'%args.feature
y_train = data_train[label].astype('int') 
y_train = y_train.reshape(np.shape(y_train)[0])
y_test = data_test[label].astype('int')
y_test = y_test.reshape(np.shape(y_test)[0])

#create dataloder
train_dataset = DataFrameDataset(data_input = X_train, label_input = y_train)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batchSize, shuffle=True)

test_dataset = DataFrameDataset(data_input = X_test, label_input = y_test)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batchSize, shuffle=True)

if args.feature == 'alcoholism':
    num_classes = 2
elif args.feature == 'stimulus':
    num_classes = 5
elif args.feature == 'id':
    num_classes = 122
else:
    raise ValueError("feature [%s] not recognized." % args.feature)

#### load specified model ####
print('==> Building model..')

if args.model == 'ResNet18':
    net = ResNet18(num_classes)
elif args.model == 'ResNet34':
    net = ResNet34(num_classes)
elif args.model == 'ResNet50':
    net = ResNet50(num_classes)
elif args.model == 'ResNet101':
    net = ResNet101(num_classes)    
elif args.model == 'ResNet152':
    net = ResNet152(num_classes)  
else:
    raise NotImplementedError('model [%s] not implemented.' % args.model)

net = net.to(device)
if device == 'cuda':
    net = torch.nn.DataParallel(net)
    cudnn.benchmark = True

best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch    
    
if args.resume:
    # Load checkpoint.
    print('==> Resuming from checkpoint..')
    assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
    checkpoint = torch.load('./checkpoint/%s/%s/ckpt.pth'%(args.feature, args.model))
    net.load_state_dict(checkpoint['net'])
    best_acc = checkpoint['acc']
    start_epoch = checkpoint['epoch']

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=args.lr,
                      momentum=0.9, weight_decay=5e-4)

'''
train the model
'''
def train(epoch):
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(train_loader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, targets.long())
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

    print('Training Loss: %.3f | Acc: %.3f%% (%d/%d)'
                     % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))

'''
test the model
'''
def test(epoch, args):
    global best_acc
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    y_pred = []
    y_target = []
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(test_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets.long())

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            y_target.append(targets)
            y_pred.append(predicted)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
        y_target = torch.cat(y_target).cpu()
        y_pred = torch.cat(y_pred).cpu() 
        
        # alcoholism detection task has 2 additonal criterial in addition to accuracy
        if args.feature == 'alcoholism':
            sensitivity = recall_score(y_target, y_pred, pos_label=1)*100.
            specificity = recall_score(y_target, y_pred, pos_label=0)*100.
            #accuracy = 100.* correct/total
            
            #the test dataset is highly imbalanced so far, therefore using the average of spec. and sens., will change later
            accuracy = (sensitivity + specificity) / 2 #* ((sensitivity - specificity)>=0)
            #print results
            print('Testing Loss: %.3f | Acc: %.3f%% | Sensitivity: %.3f%% | Specificity: %.3f%%'
                         % (test_loss/(batch_idx+1), accuracy, sensitivity, specificity))
        else:
            accuracy = 100.* correct/total
            print('Testing Loss: %.3f | Acc: %.3f%% (%d/%d)'
                         % (test_loss/(batch_idx+1), accuracy, correct, total))

    # Save checkpoint.
    if accuracy > best_acc:
        best_acc = accuracy
        #if best_acc > 85:
        print('Saving..')
        state = {
            'net': net.state_dict(),
            'acc': accuracy,
            'epoch': epoch,
        }
        if not os.path.isdir('resnet_checkpoints/%s/%s'%(args.feature, args.model)):
            os.makedirs('resnet_checkpoints/%s/%s'%(args.feature, args.model))
        #save checkpints separately, with verbose name
        torch.save(state, './resnet_checkpoints/%s/%s/ckpt_e%s_%d.pth'%(args.feature, args.model, epoch, accuracy))
        #save the best model, with the name 'ckpt.pth'
        torch.save(state, './resnet_checkpoints/%s/%s/ckpt.pth'%(args.feature, args.model))

for epoch in range(start_epoch, 200):
    train(epoch)
    test(epoch, args)
