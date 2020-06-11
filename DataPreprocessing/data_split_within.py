import scipy.io as sio
import scipy.io
import numpy as np
from sklearn.model_selection import train_test_split

'''
within-subject data spliting
'''

mat = sio.loadmat('ucieeg.mat')


data = mat['X'].astype('float32')

#data = np.transpose (data, (0, 3 ,2, 1))

print (np.shape (data))

label_alcoholism = mat['y_alcoholic'].astype('int') 
label_alcoholism = label_alcoholism.reshape(np.shape(data)[0])

label_stimulus = mat['y_stimulus'].astype('int')- 1  #label start from 0
label_stimulus = label_stimulus.reshape(np.shape(data)[0]) #label start from 0

label_id = mat['subjectid'].astype('int') - 1
label_id = label_id.reshape(np.shape (data)[0])


train_data = []
train_label = []

train_cyc_data = []
train_cyc_label = []

val_data = []
val_label = []

test_data = []
test_label = []

num_subject = 122

#loop through each subject to split data within subject
for i in range(num_subject):
    index_i = np.where(label_id == i)
    data_i = data[index_i]
    #print (np.shape (data_i))

    label_alcoholism_i = label_alcoholism[index_i]
    label_stimulus_i = label_stimulus[index_i]
    label_id_i = label_id[index_i]

    #print (np.shape(data_i))
    
    # try 8-1-1 train-test-validation splitting
    label_stack_i =  np.stack((label_alcoholism_i, label_stimulus_i, label_id_i), axis=1)
    X_train_i, X_test_i, y_train_i, y_test_i = train_test_split(data_i, label_stack_i, test_size=0.2,random_state = 1)
    X_test_i, X_val_i, y_test_i, y_val_i = train_test_split(X_test_i, y_test_i, test_size=0.5,random_state = 1)

    # if alcoholic, sample 70% data for the cyclegan-based model training => balanced data
    if label_alcoholism_i[0] == 0 or (label_alcoholism_i[0] == 1 and np.random.rand()<=0.7):
        train_cyc_data.append(X_train_i)
        
    #    X_train_i, X_test_i, y_train_i, y_test_i = train_test_split(data_i, label_stack_i, test_size=0.45,random_state = 1)
    #    X_test_i, X_val_i, y_test_i, y_val_i = train_test_split(X_test_i, y_test_i, test_size=0.33,random_state = 1)

    #else:
    #    X_train_i, X_test_i, y_train_i, y_test_i = train_test_split(data_i, label_stack_i, test_size=0.1,random_state = 1)
    #    X_test_i, X_val_i, y_test_i, y_val_i = train_test_split(X_test_i, y_test_i, test_size=0.5,random_state = 1)

    train_data.append (X_train_i)

    val_data.append (X_val_i)

    test_data.append (X_test_i)

    train_label.append (y_train_i)

    test_label.append (y_test_i)

    val_label.append (y_val_i)
    

train_data = np.concatenate(train_data)
train_label = np.concatenate(train_label)

val_data = np.concatenate(val_data)
val_label = np.concatenate(val_label)

test_data = np.concatenate(test_data)
test_label = np.concatenate(test_label)



print (np.shape (train_label))
print (np.shape (val_label))
print (np.shape (test_label))



sio.savemat( 'uci_eeg_train_within_8.mat',
            {'data':  train_data, 'label_alcoholism':np.reshape(train_label[:,0],(-1,1)), 
             'label_stimulus': np.reshape(train_label[:,1],(-1,1)), 'label_id':np.reshape(train_label[:,2],(-1,1))})

sio.savemat( 'uci_eeg_test_within_1.mat',
            {'data':  test_data, 'label_alcoholism':np.reshape(test_label[:,0],(-1,1)), 
             'label_stimulus':np.reshape(test_label[:,1],(-1,1)), 'label_id':np.reshape(test_label[:,2],(-1,1))})

sio.savemat( 'uci_eeg_validation_within_1.mat',
            {'data':  val_data, 'label_alcoholism':np.reshape(val_label[:,0],(-1,1)), 
             'label_stimulus': np.reshape(val_label[:,1],(-1,1)), 'label_id':np.reshape(val_label[:,2],(-1,1))})


