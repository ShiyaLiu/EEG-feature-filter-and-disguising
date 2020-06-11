import scipy.io as sio
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D 
from matplotlib.axes import Axes
import os

'''
Generate some sample images or plots for the report
'''

def time_freq_signals(root_path):
    data_time = sio.loadmat('uci_eeg_train_within.mat')
    data_spectrum = sio.loadmat('eeg_spectrum_train_within.mat')

    eeg_time_sample = data_time['data'][0][:,0]
    eeg_spectrum_sample = data_spectrum['data'][0][0]

    plt.figure()
    plt.plot(np.linspace(0,1,256), eeg_time_sample)
    plt.xlabel('Time (s)', fontsize=16)
    plt.ylabel('Amplitude (mV)', fontsize=16)
    plt.tick_params(axis='both', which='major', labelsize=12)
    plt.savefig('%s\EEG_time_series.png'%root_path, bbox_inches='tight')
    
    plt.figure()
    plt.plot(eeg_spectrum_sample)
    plt.xlabel('Frequency (Hz)', fontsize=16)
    plt.ylabel('Normalised Amplitude (mV)', fontsize=14)
    plt.tick_params(axis='both', which='major', labelsize=12)
    plt.savefig('%s\EEG_spectum.png'%root_path, bbox_inches='tight')


def spectrum_simple(root_path, n):
    data_spectrum = sio.loadmat('eeg_spectrum_train_within.mat')
    eeg_spectrum_sample = data_spectrum['data'][0]

    for i in range(n):
        ch = eeg_spectrum_sample[i]
        fig = plt.figure()
        plt.xticks([])
        plt.yticks([])
        plt.plot(ch)
        plt.savefig('%s\EEG_spectum_%d.png'%(root_path, i), bbox_inches='tight')


def dummy_EEG_img(root_path, i):
    data = sio.loadmat('eeg_dummy_images_w_label_step3_within.mat')
    
    images = data['data']
      
    img_in = images[i,:,:,:]
    img_in -= np.min(img_in)
    img_in /= np.max(img_in)
    
    plt.clf()
    plt.subplot(1,1,1)
    plt.axis('off')
    plt.imshow(img_in)
    plt.savefig('%s\EEG_image_dummy_sample.png'%root_path, bbox_inches='tight')


def locations_3D(root_path):
    locs = sio.loadmat('Neuroscan_locs_orig.mat')
    locs_3d = locs['A']
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')    
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_zticklabels([])
    ax.scatter(locs_3d[:,0],locs_3d[:,1],locs_3d[:,2])
    plt.savefig('%s\loc3D.png'%root_path, bbox_inches='tight')
    
if __name__ == '__main__':
    img_folder = 'images'
    if not os.path.isdir(img_folder):
        os.makedirs(img_folder)
    time_freq_signals(img_folder)
    spectrum_simple(img_folder, 3)
    dummy_EEG_img(img_folder, 11)
    locations_3D(img_folder)
    
