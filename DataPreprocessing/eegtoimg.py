from __future__ import print_function
import time

import numpy as np
np.random.seed(1234)
from functools import reduce
import math as m

import scipy.io

from scipy.interpolate import griddata
from sklearn.preprocessing import scale
from utils import augment_EEG, cart2sph, pol2cart
import scipy.io as sio

import matplotlib.pyplot as plt

from sklearn import preprocessing
#import mne
#from plotly import tools
#from plotly.graph_objs import Layout, YAxis, Scatter, Annotation, Annotations, Data, Figure, Marker, Font
#import matplotlib.pyplot as plt
#import plotly.plotly as py
#import plotly.offline as offline
#import plotly.graph_objs as go

#since it is my personal account for plotly, do not use it too often. QAQ
#tools.set_credentials_file(username='yy.fy.cn',api_key='s4aYwqqYayPPjcqFxk7p')

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

def get_fft(snippet):
    Fs = 256.0;  # sampling rate
    #Ts = len(snippet)/Fs/Fs; # sampling interval
    snippet_time = len(snippet)/Fs
    Ts = 1.0/Fs; # sampling interval
    t = np.arange(0,snippet_time,Ts) # time vector

    # ff = 5;   # frequency of the signal
    # y = np.sin(2*np.pi*ff*t)
    y = snippet
#     print('Ts: ',Ts)
#     print(t)
#     print(y.shape)
    n = len(y) # length of the signal
    k = np.arange(n)
    T = n/Fs
    frq = k/T # two sides frequency range
    frq = frq[range(int(n/2))] # one side frequency range

    Y = np.fft.fft(y)/n # fft computing and normalization
    Y = Y[range(int(n/2))]
    #Added in: (To remove bias.)
    #Y[0] = 0
    return frq,abs(Y)

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
    print('df shape',np.shape(df))

    for i in range(0, np.shape(df)[0]):
        frame = []

        for channel in range(0, np.shape(df)[1]):
            snippet = df[i][channel]
            #print(i, channel)
            f,Y =  get_fft(snippet)
            #print (Y.shape)
            theta, alpha, beta = theta_alpha_beta_averages(f,Y)
            #print (theta, alpha, beta)
            frame.append([theta, alpha, beta])
            
        frames.append(frame)
    return np.array(frames)

'''
def visual_one_eeg (images, img_path):

    #images = np.reshape(images, [self.input_height, self.input_width])
    n_channels = np.shape(images)[0]
    step = 1. / n_channels
    data = images
    eeg_time = np.linspace(1,n_channels,n_channels)
    kwargs = dict(domain=[1 - step, 1], showticklabels=False, zeroline=False, showgrid=False)

    # create objects for layout and traces
    layout = Layout(yaxis=YAxis(kwargs), showlegend=False)
    traces = [Scatter(x=eeg_time, y=data.T[:, 0])]

    # loop over the channels
    for ii in range(1, n_channels):
            kwargs.update(domain=[1 - (ii + 1) * step, 1 - ii * step])
            layout.update({'yaxis%d' % (ii + 1): YAxis(kwargs), 'showlegend': False})
            traces.append(Scatter(x=eeg_time, y=data.T[:, ii], yaxis='y%d' % (ii + 1)))

    # add channel names using Annotations

    # set the size of the figure and plot it
    layout.update(autosize=False, width=1000, height=600)
    fig = Figure(data=Data(traces), layout=layout)
    #py.image.save_as(fig, filename='')
    py.image.save_as(fig, filename=img_path)
'''

if __name__ == '__main__':
    from utils import reformatInput

    # Load electrode locations
    print('Loading data...')
    locs = scipy.io.loadmat('Neuroscan_locs_orig.mat')
    locs_3d = locs['A']
    locs_2d = []
    # Convert to 2D
    for e in locs_3d:
        locs_2d.append(azim_proj(e))
        
        
        
    for s in ('train','test','validation'):
        mat = sio.loadmat('uci_eeg_'+ s +'_within.mat')
        
        data = mat['data']
        label_alcoholic = mat['label_alcoholism']
        label_stimulus = mat['label_stimulus']
        label_id = mat['label_id']
        
        tras_X = np.transpose(data, (0, 2, 1))
        print (np.shape(tras_X))
        #visual_one_eeg (tras_X[0], "One-eeg-plot.png")
        
        X = make_frames (tras_X, 1)
        #print (np.shape(X))
        X_1 = X.reshape (np.shape(X)[0], np.shape(X)[1] * np.shape(X)[2])
        
    
        print('Generating images...')
    
        images = gen_images(np.array(locs_2d),
                                      X_1,
                                      32, normalize=False)
    
        print(images.shape)
        
        images = np.transpose (images, (0, 3 ,2, 1))
        
        ########################### add normazation #################################
        #trs_data = images.reshape (np.shape(images)[0], np.shape(images)[1] * np.shape(images)[2] * np.shape(images)[3]) 
    
        #max_abs_scaler = preprocessing.MaxAbsScaler()
    
        #trs_data = max_abs_scaler.fit_transform (trs_data)
    
        #real_data = trs_data.reshape (np.shape(images)[0], np.shape(images)[1], np.shape(images)[2], np.shape(images)[3]) 
    
        #images = real_data
        
        ###########################################################################################
            
        sio.savemat( 'uci_eeg_images_'+ s +'_within.mat',
                {'data': images, 'label_alcoholism': label_alcoholic, 'label_stimulus': label_stimulus, 'label_id': label_id})
    
        print (np.max(images))
        print (np.min(images))
          
        img_in = images[0,:,:,:]
        img_in -= np.min(img_in)
        img_in /= np.max(img_in)
    
        plt.clf()
        plt.subplot(1,1,1)
        plt.imshow(img_in)
        #plt.pause(50)
    
    '''
    print (np.shape(images))
    plt.clf()
    plt.subplot(1,1,1)
    plt.imshow(images[0])
    plt.pause(20)
    '''
    print('\n')

    