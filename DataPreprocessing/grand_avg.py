from __future__ import print_function
import time

import scipy.io as sio
import matplotlib.pyplot as plt
import numpy as np

np.random.seed(1234)
from functools import reduce
import math as m

from scipy.interpolate import griddata
from sklearn.preprocessing import scale
from utils import augment_EEG, cart2sph, pol2cart

from sklearn import preprocessing

'''
generate dummy identities using grand average
'''


'''
helper function for EEG2Img, 3D location projected to 2D plane
code from Yao
'''
def azim_proj(pos):
    """
    Computes the Azimuthal Equidistant Projection of input point in 3D Cartesian Coordinates.
    Imagine a plane being placed against (tangent to) a globe. If
    a light source inside the globe projects the graticule onto
    the plane the result would be a planar, or azimuthal, map
    projection.

    :param pos: position in 3D Cartesian coordinates
    :return: projected coordinates using Azimuthal Equidistant Projection
    """
    [r, elev, az] = cart2sph(pos[0], pos[1], pos[2])
    return pol2cart(az, m.pi / 2 - elev)

'''
helper function for EEG2Img, generate EEG images
code from Yao
'''
def gen_images(locs, features, n_gridpoints, normalize=True,
               augment=False, pca=False, std_mult=0.1, n_components=2, edgeless=False):
    """
    Generates EEG images given electrode locations in 2D space and multiple feature values for each electrode

    :param locs: An array with shape [n_electrodes, 2] containing X, Y
                        coordinates for each electrode.
    :param features: Feature matrix as [n_samples, n_features]
                                Features are as columns.
                                Features corresponding to each frequency band are concatenated.
                                (alpha1, alpha2, ..., beta1, beta2,...)
    :param n_gridpoints: Number of pixels in the output images
    :param normalize:   Flag for whether to normalize each band over all samples
    :param augment:     Flag for generating augmented images
    :param pca:         Flag for PCA based data augmentation
    :param std_mult     Multiplier for std of added noise
    :param n_components: Number of components in PCA to retain for augmentation
    :param edgeless:    If True generates edgeless images by adding artificial channels
                        at four corners of the image with value = 0 (default=False).
    :return:            Tensor of size [samples, colors, W, H] containing generated
                        images.
    """
    feat_array_temp = []
    nElectrodes = locs.shape[0]     # Number of electrodes
    # Test whether the feature vector length is divisible by number of electrodes
    assert features.shape[1] % nElectrodes == 0
    n_colors = int(features.shape[1] / nElectrodes)
    for c in range(n_colors):
        feat_array_temp.append(features[:, c * nElectrodes : nElectrodes * (c+1)])
    if augment:
        if pca:
            for c in range(n_colors):
                feat_array_temp[c] = augment_EEG(feat_array_temp[c], std_mult, pca=True, n_components=n_components)
        else:
            for c in range(n_colors):
                feat_array_temp[c] = augment_EEG(feat_array_temp[c], std_mult, pca=False, n_components=n_components)
    nSamples = features.shape[0]
    # Interpolate the values
    grid_x, grid_y = np.mgrid[
                     min(locs[:, 0]):max(locs[:, 0]):n_gridpoints*1j,
                     min(locs[:, 1]):max(locs[:, 1]):n_gridpoints*1j
                     ]
    temp_interp = []
    for c in range(n_colors):
        temp_interp.append(np.zeros([nSamples, n_gridpoints, n_gridpoints]))
    # Generate edgeless images
    if edgeless:
        min_x, min_y = np.min(locs, axis=0)
        max_x, max_y = np.max(locs, axis=0)
        locs = np.append(locs, np.array([[min_x, min_y], [min_x, max_y],[max_x, min_y],[max_x, max_y]]),axis=0)
        for c in range(n_colors):
            feat_array_temp[c] = np.append(feat_array_temp[c], np.zeros((nSamples, 4)), axis=1)
    # Interpolating
    for i in range(nSamples):
        for c in range(n_colors):
            temp_interp[c][i, :, :] = griddata(locs, feat_array_temp[c][i, :], (grid_x, grid_y),
                                    method='cubic', fill_value=np.nan)
        print('Interpolating {0}/{1}\r'.format(i+1, nSamples), end='\r')
    
    # Normalizing
    for c in range(n_colors):
        if normalize:
            temp_interp[c][~np.isnan(temp_interp[c])] = \
                scale(temp_interp[c][~np.isnan(temp_interp[c])])
        temp_interp[c] = np.nan_to_num(temp_interp[c])    #Replace nan with zero and inf with large finite numbers.
    return np.swapaxes(np.asarray(temp_interp), 0, 1)     # swap axes to have [samples, colors, W, H]

'''
helper function for EEG2Img, obtain features of 3 frequency band by averaging spectrum attitude
code from Yao
'''
def theta_alpha_beta_averages(f,Y):
    theta_range = (4,8)
    alpha_range = (8,13)
    beta_range = (13,30)
    theta = Y[(f>theta_range[0]) & (f<=theta_range[1])].mean()
    alpha = Y[(f>alpha_range[0]) & (f<=alpha_range[1])].mean()
    beta = Y[(f>beta_range[0]) & (f<=beta_range[1])].mean()
    return theta, alpha, beta


def make_frames(df,frame_duration):
    '''
    in: dataframe or array with all channels, frame duration in seconds
    out: array of theta, alpha, beta averages for each probe for each time step
        shape: (n-frames,m-probes,k-brainwave bands)
    '''
    Fs = 256.0
    frame_length = Fs*frame_duration
    
    frames = []
    #print('df shape',np.shape(df))
    for i in range(0, np.shape(df)[0]):
        frame = []
        
        for channel in range(0, np.shape(df)[1]):
            snippet = df[i][channel]
            #print(i, channel)
            #f,Y =  get_fft(snippet)
            #print (len(snippet))
            theta, alpha, beta = theta_alpha_beta_averages(np.array(range(len(snippet))), snippet)
            #print (theta, alpha, beta)
            frame.append([theta, alpha, beta])
            
        frames.append(frame)
        if i % 100 == 0:
            print('===== %d end ====='%(i))
    return np.array(frames)



if __name__ == '__main__':
    
    #load EEG spectrums in training set
    data = sio.loadmat('eeg_spectrum_train_within.mat') 
    
    
    label_disease_range = 2
    label_stimulus_range = 5
    num_ch, band_width = data['data'][0].shape
    num_subjects = 120
    
    
    ####### Group the EEG spectrums #######
    #generate 10 groups: 10 combinations of alcoholism and stimulus attributes
    groups = [{} for i in range(label_disease_range * label_stimulus_range)]    #map: subject id -> data
    
    #allocate each trial of EEG signals to the corresponding group
    for i in range(len(data['data'])):
        index_label = label_stimulus_range * data['label_alcoholism'][i,0] + data['label_stimulus'][i,0]
        index_id = data['label_id'][i,0]
        
        #add EEG data to the corresponding group and subject 
        if index_id in groups[index_label]:
            groups[index_label][index_id].append(data['data'][i])
        else:
            groups[index_label][index_id] = [data['data'][i]]

    ####### visulise #######
    #g0 = groups[0]
    #s = [g0[121][3][17], g0[70][3][17], g0[72][3][17]]
    #cnt = 0;
    #for c in s:
    #    plt.figure()
    #    plt.xticks([])
    #    plt.yticks([])
    #    plt.plot(c)
    #    plt.savefig('EEG_spectum_candi_%d.png'%cnt, bbox_inches='tight')
    #    cnt += 1
    #avg = np.mean(s,axis=0)
    #plt.figure()
    #plt.xticks([])
    #plt.yticks([])
    #plt.plot(avg)
    #plt.savefig('EEG_spectum_candi_avg.png', bbox_inches='tight')
    #cnt += 1    
    
    ####### generate labellled EEG spectrums with dummy identities #######
    dummy = []
    dummy_label_disease = []
    dummy_label_stimulus = []
    
    #loop through 10 groups
    for i in range(len(groups)):
        #sort each group w.r.t the number of EEG signals for each subject
        candidate_list = sorted(list(groups[i].values()), key=len)
        step = 3    #a sliding window of 3
        #average trials of EEG siganls of the subjects within the window
        for j in range(len(candidate_list) - step + 1):
            for k in range(len(candidate_list[j])):
                #average across subjects
                new = np.mean([item[k] for item in candidate_list[j:j+step]], axis=0)
                dummy.append(new)
                #label the dunmmy identity
                dummy_label_disease.append(i//label_stimulus_range)
                dummy_label_stimulus.append(i%label_stimulus_range)
                
    dummy = np.array(dummy)
 
    
    #######  generate EEG images with dummy identities #######
    # Load electrode locations
    print('Loading data...')
    locs = sio.loadmat('Neuroscan_locs_orig.mat')
    locs_3d = locs['A']
    locs_2d = []
    # Convert to 2D
    for e in locs_3d:
        locs_2d.append(azim_proj(e))
    
    X = make_frames(dummy, 1)
    #print (np.shape(X))
    X_1 = X.reshape(np.shape(X)[0], np.shape(X)[1] * np.shape(X)[2])
    

    print('Generating images...')

    images = gen_images(np.array(locs_2d),
                                  X_1,
                                  32, normalize=False)

    images = np.transpose (images, (0, 3 ,2, 1))
    
    # save the dummy EEG images
    sio.savemat( 'eeg_dummy_images_w_label_step3_within.mat',
            {'data': images,'label_alcoholism':np.reshape(dummy_label_disease,(-1,1)),
             'label_stimulus':np.reshape(dummy_label_stimulus,(-1,1))})



    
    # print (np.max(images))
    # print (np.min(images))
      
    # img_in = images[0,:,:,:]
    # img_in -= np.min(img_in)
    # img_in /= np.max(img_in)

    # plt.clf()
    # plt.subplot(1,1,1)
    # plt.imshow(img_in)
    # #plt.pause(50)

    # print (np.shape(images))
    # plt.clf()
    # plt.subplot(1,1,1)
    # plt.imshow(images[0])
    # plt.pause(20)
    
    
    #print('\n')
            
    #print(spectrums.shape)
    #sample0 = np.array(data['X'][:88]).mean(axis=0)
    
    #sample_n = data['X'][6001]
    
    #plt.plot(sample_n)
    
