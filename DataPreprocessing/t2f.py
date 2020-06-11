import scipy.io as sio
import matplotlib.pyplot as plt
import numpy as np

'''
time-frequency convertion using FFT
'''
def get_fft(snippet):
    Fs = 256.0;  # sampling rate
    #Ts = len(snippet)/Fs/Fs; # sampling interval
    #snippet_time = len(snippet)/Fs
    #Ts = 1.0/Fs; # sampling interval
    #t = np.arange(0,snippet_time,Ts) # time vector

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



if __name__ == '__main__':
    
    data = sio.loadmat('uci_eeg_train_within.mat') 
    
    X = np.transpose(data['data'], (0, 2, 1))
    
    num_exp, num_ch, rate = X.shape
    
    spectrums = []
    
    # time-frequency convertion on all the EEG signals
    for i in range(num_exp):
        spectrum = []
        for ch in range(num_ch):
            time_domain = X[i][ch]
            f , magnitude = get_fft(time_domain)
            spectrum.append(magnitude)
        spectrums.append(spectrum)
        
    dummy = np.array([])
    
    sio.savemat('eeg_spectrum_train_within.mat', {'data': spectrums, 'label_alcoholism':data['label_alcoholism'], 
                                'label_stimulus':data['label_stimulus'], 'label_id':data['label_id']})
    
    
    
 