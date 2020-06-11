import random
import os.path
from data.base_dataset import BaseDataset
import scipy.io as sio
import numpy as np


class EEGDataset(BaseDataset):
    def name(self):
        return 'EEGDataset'

    def initialize(self, opt):
        self.opt = opt
        self.root = opt.dataroot

        mat_eeg_ori = sio.loadmat(self.root + opt.data_real)
        mat_eeg_dummy = sio.loadmat(self.root + opt.data_dummy)

        self.data_ori = np.transpose(mat_eeg_ori['data'],(0,3,2,1))
        #self.data_ori = mat_eeg_ori['data']
        self.label_alc_ori = mat_eeg_ori['label_alcoholism']
        self.label_stimulus_ori = mat_eeg_ori['label_stimulus']
        self.label_id_ori = mat_eeg_ori['label_id']
        self.label_combined_ori = mat_eeg_ori['label_combined']


        
        self.data_dummy = np.transpose(mat_eeg_dummy['data'], (0,3,2,1))
        self.label_alc_dummy = mat_eeg_dummy['label_alcoholism']
        self.label_stimulus_dummy = mat_eeg_dummy['label_stimulus']
        self.label_combined_dummy = mat_eeg_dummy['label_combined']

        #print (np.shape (self.data_dummy ))
        #print (np.shape (mat_eeg['data_normal']))
        #print (np.shape (self.data_disease))
        #print (type(self.label_normal))
        #print (np.shape (self.label_normal))
        #print (np.shape (self.label_disease))
        self.shuffle_indices()
        #assert (0)

    def shuffle_indices(self):

        self.ori_indices = list(range(self.data_ori.shape[0]))
        self.dummy_indices = list(range(self.data_dummy.shape[0]))
        print('num original EEG images : ', len(self.ori_indices), 'num dummy EEG images : ', len(self.dummy_indices))
        if not self.opt.serial_batches:
            random.shuffle(self.ori_indices)
            random.shuffle(self.dummy_indices)

    def __getitem__(self, index):

        if index == 0:
            self.shuffle_indices()
        
        #if random.random() < 0.5: # invert the color with 50% prob

        O_img = self.data_ori[self.ori_indices[index % len(self.ori_indices)]].astype('float32')#.ToTensor()
        #print("label_id:",self.label_id_ori.shape)
        #print('index:',len(self.ori_indices))
        O_label_id = self.label_id_ori[self.ori_indices[index % len(self.ori_indices)]][0]
        O_label_alc = self.label_alc_ori[self.ori_indices[index % len(self.ori_indices)]][0]
        O_label_stimulus = self.label_stimulus_ori[self.ori_indices[index % len(self.ori_indices)]][0]
        O_label_combined = self.label_combined_ori[self.ori_indices[index % len(self.ori_indices)]][0]


        #D_img = self.transform(D_img)
        D_img = self.data_dummy[self.dummy_indices[index % len(self.dummy_indices)]].astype('float32')#.ToTensor()
        D_label_alc = self.label_alc_dummy[self.dummy_indices[index % len(self.dummy_indices)]][0]
        D_label_stimulus = self.label_stimulus_dummy[self.dummy_indices[index % len(self.dummy_indices)]][0]
        D_label_combined = self.label_combined_dummy[self.dummy_indices[index % len(self.dummy_indices)]][0]        
     
        item = {}
        item.update({'A': D_img,
                     #'A_paths': A_path,
                     'A_label_alcoholism': D_label_alc,
                     'A_label_stimulus': D_label_stimulus,
                     'A_label_combined': D_label_combined                                         
                 })
        
        item.update({'B': O_img,
                     #'B_paths': B_path,
                     'B_label_id': O_label_id,
                     'B_label_alcoholism': O_label_alc,
                     'B_label_stimulus': O_label_stimulus, 
                     'B_label_combined': O_label_combined,                                        
                 })
        return item
        
    def __len__(self):

        return len(self.ori_indices)
        
