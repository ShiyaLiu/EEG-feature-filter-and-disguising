
import numpy as np
#from SyncNetModel import SyncNetModel
import scipy.io
import os
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.utils.data as Data
import torchvision
from mpl_toolkits.mplot3d import Axes3D
import torch.nn.functional as F
#import utils

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
from sklearn import preprocessing
import time
import scipy.io as sio
import scipy.io
import os
from imblearn.under_sampling import NearMiss 
from collections import Counter

from sklearn.datasets import make_classification
from imblearn.under_sampling import NearMiss 
from Image_wise_autoencoders import CNN
EPOCH = 10000
BATCH_SIZE = 64
num_epoch = 60


class DataFrameDataset_train(torch.utils.data.Dataset):
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

class ClassificationNet(nn.Module):
    def __init__(self):
        super().__init__()
        
        #self.fc1 = torch.nn.Linear(in_features = 1024, out_features = 256)
        #self.fc2 = torch.nn.Linear(in_features = 256,  out_features = 2)
        

        self.fc1 = torch.nn.Linear(in_features = 1024, out_features = 512)
        self.fc2 = torch.nn.Linear(in_features = 512,  out_features = 128)
        self.fc3 = torch.nn.Linear(in_features = 128,  out_features = 5)

        self.dropout1 = nn.Dropout(p=0.1)
        self.dropout2 = nn.Dropout(p=0.1)
        
    def forward(self, x):
        
        x = x.view(x.size(0), -1)
        #x = self.dropout1(x)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout1(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc3(x)
        return x

if __name__ == '__main__':

    data_root = '../datasets/eeg/'
    data_train = sio.loadmat(data_root+'eeg_images_train_augmented_within.mat')
    data_test = sio.loadmat(data_root+'uci_eeg_images_test_within.mat')

    X_train = np.transpose(data_train['data'],(0,3,2,1)).astype('float32')
    X_test = np.transpose(data_test['data'],(0,3,2,1)).astype('float32')
    
    y_train = data_train['label_stimulus'].astype('int') 
    y_train = y_train.reshape(np.shape(y_train)[0])
    y_test = data_test['label_stimulus'].astype('int')
    y_test = y_test.reshape(np.shape(y_test)[0])
    
    train_dataset = DataFrameDataset_train(data_input = X_train, label_input = y_train)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

    test_dataset = DataFrameDataset_train(data_input = X_test, label_input = y_test)

    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=True)

    Image_AC = CNN()
    Image_AC.cuda()

    criterian = torch.nn.MSELoss()
    learning_rate = 1e-4
    optimizer = torch.optim.Adam(Image_AC.parameters(), weight_decay=1e-4 ,lr = learning_rate)
    print (Image_AC)
    #nn.load_state_dict(torch.load('Image-wise_autoencoders.pkl'))

    for i in range(num_epoch):
        #break
        #training 
        Image_AC.train()
        for img, _ in train_loader:
            #print (type (img))
            ##print (np.shape (img))
            #print (np.shape(_))
            #assert (0)
            img = Variable(img).cuda()

            Image_AC.zero_grad()
            output = Image_AC(img, False)
            #print (np.shape(output[0]))
            loss = criterian(output[1], img)

            loss.backward()
            optimizer.step()

        #get train loss
        train_loss = 0
        train_correct = 0
        Image_AC.eval()
        for img, _ in train_loader:

            img = Variable(img).cuda()

            output = Image_AC(img)
            train_loss += criterian(output[1], img)#.data[0]

        print("autoencoder epoch : ",i, "Train: Loss: %.6f" %(train_loss/len(train_dataset)))

        ################################################## ClassificationNet ###############################################
    #print ("writing pkf for image wise autoencoder")
   
    net = ClassificationNet().cuda()

    # Loss and Optimizer
    criterion = nn.CrossEntropyLoss(size_average=False)
    learning_rate = 4e-5
    optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)

    learning_rate = 1e-7
    optimizer_AC = torch.optim.Adam(Image_AC.parameters(), lr=learning_rate)


    max_acc = 0
    min_loss = 100

    for i in range(EPOCH):
        net.train()
        Image_AC.train()
        #training 
        for signal, label in train_loader:
            #print (np.shape(signal)[0])
            signal = Variable(signal).cuda()
            signal = Image_AC(signal)[0]
            
            #print ()
            #print (label)
            #print (np.shape(label))
            label = Variable(label).cuda()
            label = label.long().view(label.size(0))
            net.zero_grad()
            Image_AC.zero_grad()
            output = net(signal)
            loss = criterion(output, label)
            _, predict = torch.max(output, 1)
        
            loss.backward()
            optimizer.step()
            optimizer_AC.step()

            train_loss = 0
            train_correct = 0
            #break

        #break
        #get train los
        for signal, label in train_loader:
            #print (np.shape(signal))
            #print ("asddas")
            #print (np.shape(signal))
            #print (np.shape(label))
            signal = Variable(signal).cuda()
            signal = Image_AC(signal)[0]
            label = Variable(label).cuda()
            label = label.long().view(label.size(0))
            output = net(signal)
            #print ()
            train_loss += criterion(output, label)#.data[0]

            _, predict = torch.max(output, 1)
            #print (predict)
            train_correct += np.sum((predict == label).data.cpu().numpy())
        
        #get test loss and accuracy
        print(i, "Train: Loss: %.5f, Acc: %.4f" %(train_loss/len(train_dataset), train_correct/len(train_dataset)))
        test_loss = 0
        test_correct = 0
        cnt = 0
        Image_AC.eval()
        net.eval()
        for signal, label in test_loader:
            signal = Variable(signal).cuda()
            signal = Image_AC(signal)[0]
            label = Variable(label).cuda()
            label = label.long().view(label.size(0))
            output = net(signal)
            test_loss += criterion(output, label)#.data[0]

            _, predict = torch.max(output, 1)
            test_correct += np.sum((predict == label).data.cpu().numpy())
            #print (predict)
            #print (label)
            #print (np.sum((predict == label).data.cpu().numpy())/64)
            #print ("############################################################################")
            #break
            cnt += 1
        #break
        #print (cnt)
        #print (len(test_dataset))

        if test_correct/len(test_dataset) > max_acc:
            max_acc = test_correct/len(test_dataset)
            min_loss = test_loss/len(test_dataset)
            if not os.path.isdir('autoencoder_checkpoints'):
                os.makedirs('autoencoder_checkpoints')
            torch.save(net.state_dict(), 'autoencoder_checkpoints/final_classification_model_stimulus.pkl')
            torch.save(Image_AC.state_dict(), 'autoencoder_checkpoints/Image-wise_autoencoders_within_stimulus.pkl')

        print("epoch:", i, "test: Loss: %.5f, Acc: %.4f" %(test_loss/len(test_dataset), test_correct/len(test_dataset)),'best_acc' , max_acc)




