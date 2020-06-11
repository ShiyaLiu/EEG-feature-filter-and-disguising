# -*- coding: utf-8 -*-
"""
Created on Wed May 20 01:17:56 2020

@author: konaw
"""

import scipy.io as sio
import numpy as np


'''
Generate joint training set by combining the dummy EEG images with the training set 
'''

data_real = sio.loadmat('uci_eeg_images_train_within.mat')
data_dummy = sio.loadmat('eeg_dummy_images_w_label_step3_within.mat')


data = np.append(data_real['data'], data_dummy['data'],axis=0)
label_alc = np.append(data_real['label_alcoholism'], data_dummy['label_alcoholism'],axis=0)
label_stimulus = np.append(data_real['label_stimulus'], data_dummy['label_stimulus'],axis=0)

sio.savemat('eeg_images_train_augmented_within.mat',{'data':data, 'label_alcoholism':label_alc,
                                                'label_stimulus':label_stimulus})