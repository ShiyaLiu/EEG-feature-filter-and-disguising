import random
import os.path
from data.base_dataset import BaseDataset
import scipy.io as sio
import numpy as np

'''
EEG dataset only with orginal EEG images
the code is modified 
from https://github.com/jhoffman/pytorch-CycleGAN-and-pix2pix/blob/e484612d83449d05024a7d5fd2e012be53faad85/data/single_dataset.py
'''
class EEGDataset(BaseDataset):
    def name(self):
        return 'EEGDataset'

    def initialize(self, opt):
        self.opt = opt
        self.root = opt.dataroot

        mat_eeg_ori = sio.loadmat(self.root + opt.data)

        self.data = np.transpose(mat_eeg_ori['data'],(0,3,2,1))

        self.label_alcoholism = mat_eeg_ori['label_alcoholism']
        self.label_stimulus = mat_eeg_ori['label_stimulus']
        self.label_id = mat_eeg_ori['label_id']

        self.shuffle_indices()

    def shuffle_indices(self):

        self.indices = list(range(self.data.shape[0]))
        print('num testing EEG images : ', len(self.indices))
        if not self.opt.serial_batches:
            random.shuffle(self.indices)

    def __getitem__(self, index):

        if index == 0:
            self.shuffle_indices()
        
        O_img = self.data[self.indices[index % len(self.indices)]].astype('float32')#.ToTensor()
        #print("label_id:",self.label_id_ori.shape)
        #print('index:',len(self.ori_indices))
        O_label_id = self.label_id[self.indices[index % len(self.indices)]][0]
        O_label_alc = self.label_alcoholism[self.indices[index % len(self.indices)]][0]
        O_label_stimulus = self.label_stimulus[self.indices[index % len(self.indices)]][0]
     
        item = {}
        
        item.update({'O': O_img,
                     'O_label_id': O_label_id,
                     'O_label_alcoholism': O_label_alc,
                     'O_label_stimulus': O_label_stimulus                   
                 })
        return item
        
    def __len__(self):

        return len(self.indices)
        
