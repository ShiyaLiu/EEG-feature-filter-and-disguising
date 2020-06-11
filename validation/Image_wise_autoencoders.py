import torch
from torchvision import transforms, datasets
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np
import time
from scipy.spatial import distance
import math
import matplotlib.pyplot as plt
import scipy.io as sio
from sklearn.model_selection import train_test_split
import scipy.io
import pandas as pd

################################################## Hyper Parameters ###############################################
BATCH_SIZE = 64
shard_wight_judge = False
num_epoch = 60
################################################## prepare data ###############################################

transform = transforms.Compose(  
    [transforms.RandomHorizontalFlip(),
     transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]  
)  



class DataFrameDataset_train(torch.utils.data.Dataset):
    def __init__(self, data_input):
        self.data_tensor = torch.Tensor(data_input)
    # a function to get items by index
    def __getitem__(self, index):
        obj = self.data_tensor[index]
        signal = self.data_tensor[index]

        return signal

    # a function to count samples
    def __len__(self):
        n = np.shape(self.data_tensor)[0]
        #print (n)
        return n


class CNN(torch.nn.Module):

    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(in_channels = 3, out_channels = 16, kernel_size = 3, stride = 1, padding = 1)

        torch.nn.init.xavier_normal(self.conv1.weight, gain=torch.nn.init.calculate_gain('relu'))

        self.conv2 = torch.nn.Conv2d(in_channels = 16, out_channels = 16, kernel_size = 3, stride = 1, padding = 1)

        torch.nn.init.xavier_normal(self.conv2.weight, gain=torch.nn.init.calculate_gain('relu'))

        self.conv3 = torch.nn.Conv2d(in_channels = 16, out_channels = 16, kernel_size = 3, stride = 1, padding = 1)

        torch.nn.init.xavier_normal(self.conv3.weight, gain=torch.nn.init.calculate_gain('relu'))

        self.deconv1 = torch.nn.ConvTranspose2d(in_channels = 16, out_channels = 16, kernel_size = 3, stride = 1, padding = 1)

        torch.nn.init.xavier_normal(self.deconv1.weight, gain=torch.nn.init.calculate_gain('relu'))

        self.deconv2 = torch.nn.ConvTranspose2d(in_channels = 16, out_channels = 16, kernel_size = 3, stride = 1, padding = 1)

        torch.nn.init.xavier_normal(self.deconv2.weight, gain=torch.nn.init.calculate_gain('relu'))

        self.deconv3 = torch.nn.ConvTranspose2d(in_channels = 16, out_channels = 3, kernel_size = 3, stride = 1, padding = 1)

        torch.nn.init.xavier_normal(self.deconv3.weight, gain=torch.nn.init.calculate_gain('relu'))

    def forward(self, x, reconstruct = True):

        
        x = self.conv1(x)      
        x, idc1 = F.max_pool2d(x, 2, 2, return_indices=True) 
        x = F.relu(x)
        x = F.dropout2d(x, 0.25)

        x = self.conv2(x) 
        x, idc2 = F.max_pool2d(x, 2, 2, return_indices=True) 
        x = F.relu(x)
        x = F.dropout2d(x, 0.25)

        x = self.conv3(x) 
        x = F.relu(x)
        if shard_wight_judge:
            decode = F.conv_transpose2d(x, self.conv3.weight, stride=1, padding=1)  
        else:
            decode = self.deconv1(x)
        decode = F.max_unpool2d(decode, idc2, 2, 2)
        decode = F.relu(decode)
        decode = F.dropout2d(decode, 0.25)

        if shard_wight_judge:
            decode = F.conv_transpose2d(decode, self.conv2.weight, None, 1, 1)
        else:
            decode = self.deconv2(decode)
        decode = F.max_unpool2d(decode, idc1, 2, 2)
        decode = F.relu(decode)
        decode = F.dropout2d(decode, 0.25)

        if shard_wight_judge:
            decode = F.conv_transpose2d(decode, self.conv1.weight, None, 1, 1)
        else:
            decode = self.deconv3(decode)
        x = x.view(x.size(0), -1)
        #decode = F.tanh(decode)   # add in 2018.5.27
        #print (x.size())
        return x, decode

if __name__ == '__main__':
    mat = scipy.io.loadmat('eeg_images.mat')

    data = mat['X'].astype('float32')

    label = mat['label'].astype('long')

    data = np.transpose (data, (0, 3 ,2, 1))
    label = np.transpose(label)
    print (np.shape(data))
    print (np.shape(label))


    X_train, X_test, y_train, y_test = train_test_split(data, label, test_size=0.2,random_state = 1)
    print (np.shape (X_train))
    print (type(X_train))
    print (np.shape (y_train))
    print (BATCH_SIZE)
    assert(0)
    train_dataset = DataFrameDataset_train(data_input=X_train)

    train_data_loader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

    test_dataset = DataFrameDataset_train(data_input=X_test)

    test_data_loader = torch.utils.data.DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=True)

    plt.ion()
    learning_rate = 1e-4

    nn = CNN()
    nn.cuda()

    criterian = torch.nn.MSELoss()

    optimizer = torch.optim.Adam(nn.parameters(), weight_decay=1e-4 ,lr = learning_rate)



    print (nn)


    start_time = time.time()

    #nn.load_state_dict(torch.load('Image-wise_autoencoders.pkl'))

    for i in range(num_epoch):
        #break
        #training 
        nn.train()
        for img in train_data_loader:
            print (np.shape(img))
            assert (0)
            img = Variable(img).cuda()

            nn.zero_grad()
            output = nn(img, False)
            #print (np.shape(output[0]))
            loss = criterian(output[1], img)

            loss.backward()
            optimizer.step()

        #get train loss
        train_loss = 0
        train_correct = 0
        nn.eval()
        for img in train_data_loader:

            img = Variable(img).cuda()

            output = nn(img)
            train_loss += criterian(output[1], img).data[0]

        print("epoch : ",i, "Train: Loss: %.6f" %(train_loss/len(train_dataset)))

        #get test loss
        test_loss = 0
        test_correct = 0

        for img  in test_data_loader:

            img = Variable(img).cuda()

            output = nn(img)

            test_loss += criterian(output[1], img).data[0]

        print("epoch : ",i, "test: Loss: %.6f" %(test_loss/len(test_dataset)))
        
        #visulization
        '''
        img_in = img[0,:,:,:].data.cpu().numpy()
        img_in -= np.min(img_in)
        img_in /= np.max(img_in)
        img_out = output[1][0,:,:,:].data.cpu().numpy()
        img_out -= np.min(img_out)
        img_out /= np.max(img_out)

        img_in = np.transpose(img_in, (1,2,0))
        img_out = np.transpose(img_out, (1,2,0))

        plt.clf()
        plt.subplot(1,2,1)
        plt.imshow(img_in)
        plt.subplot(1,2,2)
        plt.imshow(img_out)
        plt.pause(0.1)
        '''
        #break
    print ("total_traning time", time.time() - start_time)
    final_image = []
    torch.save(nn.state_dict(), 'Image-wise_autoencoders.pkl')
    for i in range(0, np.shape(data)[0]):
        signal = data[i].reshape(1, np.shape(data[i])[0], np.shape(data[i])[1], np.shape(data[i])[2])
        data_final = Variable(torch.tensor(signal)).cuda()
        output_total= nn(data_final)
        final_image.append (output_total[0].cpu().detach().numpy())
    final_image = np.asarray(final_image).reshape(np.shape(final_image)[0], np.shape(final_image)[2])
    if shard_wight_judge == False:
        sio.savemat( 'Normal_Image-wise_autoencoders_out.mat',
                {'X': final_image, 'label': label})
    else:
        sio.savemat( 'shard_wight_Image-wise_autoencoders_out.mat',
                {'X': final_image, 'label': label})










