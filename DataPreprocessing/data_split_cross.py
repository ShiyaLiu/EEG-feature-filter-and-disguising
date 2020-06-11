import scipy.io as sio
import scipy.io
import numpy as np
from sklearn.model_selection import train_test_split

'''
cross-subject data splitting
'''

mat = scipy.io.loadmat('ucieeg.mat')


data = mat['X'].astype('float32')

#data = np.transpose (data, (0, 3 ,2, 1))

print (np.shape (data))

label_alcoholism = mat['y_alcoholic'].astype('int') 
label_alcoholism = label_alcoholism.reshape(np.shape(data)[0])

label_stimulus = mat['y_stimulus'].astype('int')- 1
label_stimulus = label_stimulus.reshape(np.shape(data)[0])

label_id = mat['subjectid'].astype('int') - 1
label_id = label_id.reshape(np.shape (data)[0])


#train_data, test_data, val_data = [], [], []
#train_label, test_label, val_label = [], [], []

num_subject = 122
num_datapoint = data.shape[0]

mask = np.zeros(num_subject)

# 7-2-1 for tarin-test-validation cross-subject data splitting 
for i in range(num_subject):
    r = np.random.rand()
    if r < 0.7:
        mask[i] = 0
    elif r >= 0.7 and r < 0.9:
        mask[i] = 1
    else:
        mask[i] = 2

#split according to subject id 
#70% of subjects will be in training set      
train_data = [data[i] for i in range(num_datapoint) if mask[label_id[i]] == 0]
train_label_alcoholism = [label_alcoholism[i] for i in range(num_datapoint) if mask[label_id[i]] == 0]
train_label_stimulus = [label_stimulus[i] for i in range(num_datapoint) if mask[label_id[i]] == 0]
train_label_id = [label_id[i] for i in range(num_datapoint) if mask[label_id[i]] == 0]

#20% subjects in testing set
test_data = [data[i] for i in range(num_datapoint) if mask[label_id[i]] == 1]
test_label_alcoholism = [label_alcoholism[i] for i in range(num_datapoint) if mask[label_id[i]] == 1]
test_label_stimulus = [label_stimulus[i] for i in range(num_datapoint) if mask[label_id[i]] == 1]
test_label_id = [label_id[i] for i in range(num_datapoint) if mask[label_id[i]] == 1]

# 10% subjects for validation set
val_data = [data[i] for i in range(num_datapoint) if mask[label_id[i]] == 2]
val_label_alcoholism = [label_alcoholism[i] for i in range(num_datapoint) if mask[label_id[i]] == 2]
val_label_stimulus = [label_stimulus[i] for i in range(num_datapoint) if mask[label_id[i]] == 2]
val_label_id = [label_id[i] for i in range(num_datapoint) if mask[label_id[i]] == 2]

#save data
sio.savemat( 'uci_eeg_train_cross.mat',
            {'data':  train_data, 'label_alcoholism':np.reshape(train_label_alcoholism,(-1,1)), 
              'label_stimulus': np.reshape(train_label_stimulus,(-1,1)), 'label_id':np.reshape(train_label_id,(-1,1))})

sio.savemat( 'uci_eeg_test_cross.mat',
            {'data':  test_data, 'label_alcoholism':np.reshape(test_label_alcoholism,(-1,1)), 
              'label_stimulus':np.reshape(test_label_stimulus,(-1,1)), 'label_id':np.reshape(test_label_id,(-1,1))})

sio.savemat( 'uci_eeg_validation_cross.mat',
            {'data':  val_data, 'label_alcoholism':np.reshape(val_label_alcoholism,(-1,1)), 
              'label_stimulus': np.reshape(val_label_stimulus,(-1,1)), 'label_id':np.reshape(val_label_id,(-1,1))})


