
# EEG feature filter and disguising 

This is our code for Neurocomputing paper Information-preserving Feature Filter for Short-term EEG signals. This feature filter can filter out alcoholism, stimulus or person identities with desired features kept. Our designed feature filter can be used for privacy protection. Our [paper](https://reader.elsevier.com/reader/sd/pii/S0925231220303349?token=F4252E1F17EBF95BE064FDFE4DE2AB4909F026FCA7574549CB521D515B679F2E30ED9B359796D413E5C160EC127AE710) are released online. 

![fig1](https://github.com/ShiyaLiu/EEG-feature-filter-and-disgusing/blob/master/imgs/filter_structure.PNG)  

### Data preprocessing
For the data preprocessing step, simply run the code listed below in order.
 1. data splitting
    - within-subject data splitting: \DataPreprocessing\data_split_within.py
    - cross-subject data spplitting: \DataPreprocessing\data_split_cross.py
 2. EEG signals to images
    - \DataPreprocessing\eegtoimg.py
### Generate EEG images with dummy identities
please run the following code in order: 
 1. time-frequency conversion: 
    - \DataPreprocessing\t2f.py
    - file generated:
 2. From EEG spectrums to EEG images with dummy identities:
    - \DataPreprocessing\grand_avg.py
 3. joint training set:
    - \DataPreprocessing\gen_joint_train_set.py
 4. the following .mat file generated before should be placed in the folder \EEG_idendity_disguising\datasets\eeg
    - eeg_dummy_images_w_label_step3_within.mat
    - eeg_images_train_augmented_within.mat
    - uci_eeg_images_test_within.mat
    - uci_eeg_images_train_within.mat
    - uci_eeg_images_validation_within.mat

**I manually did this because I run the data processing code on my own device but run my code for the model on a virtual machine.*

### Train the EEG disguising Model (feature filter)
- First use `data_extra_combined_label.ipynb` in the folder \EEG_idendity_disguising\datasets\eeg to generate training set with extra combined label
- Run the script:
```
python -m visdom.server
```
- Train the model with the semantic constraint on alocoholism feature
```
python train.py --name cycada_alcoholism --resize_or_crop=None --loadSize=32 --fineSize=32 --which_model_netD n_layers --n_layers_D 3 --model cycle_gan_semantic --lambda_A 1 --lambda_B 1 --lambda_identity 0 --no_flip --batchSize 64 --dataset_mode EEG --dataroot ./datasets/eeg/ --data_real uci_eeg_images_train_within_extra.mat --data_dummy eeg_dummy_images_w_label_step3_within_extra.mat --which_direction BtoA --feature alcoholism --num_classes 2
```
- Train the model with the semantic constraint on stimulus
```
python train.py --name cycada_stimulus --resize_or_crop=None --loadSize=32 --fineSize=32 --which_model_netD n_layers --n_layers_D 3 --model cycle_gan_semantic --lambda_A 1 --lambda_B 1 --lambda_identity 0 --no_flip --batchSize 64 --dataset_mode EEG --dataroot ./datasets/eeg/ --data_real uci_eeg_images_train_within_extra.mat --data_dummy eeg_dummy_images_w_label_step3_within_extra.mat --which_direction BtoA --feature stimulus --num_classes 5
```
- Train the model with the semantic constraint on the combined label (alcoholism + stimulus)

```
python train.py --name cycada_combined --resize_or_crop=None --loadSize=32 --fineSize=32 --which_model_netD n_layers --n_layers_D 3 --model cycle_gan_semantic --lambda_A 1 --lambda_B 1 --lambda_identity 0 --no_flip --batchSize 64 --dataset_mode EEG --dataroot ./datasets/eeg/ --data_real uci_eeg_images_train_within_extra.mat --data_dummy eeg_dummy_images_w_label_step3_within_extra.mat --which_direction BtoA --feature combined --num_classes 10
```    
- Train the model without semantic constraint
```
python train.py --name cyclegan --resize_or_crop=None --loadSize=32 --fineSize=32 --which_model_netD n_layers --n_layers_D 3 --model cycle_gan --lambda_A 1 --lambda_B 1 --lambda_identity 0 --no_flip --batchSize 64 --dataset_mode EEG --dataroot ./datasets/eeg/ --data_real uci_eeg_images_train_within_extra.mat --data_dummy eeg_dummy_images_w_label_step3_within_extra.mat --which_direction BtoA
```
### Train the Classification Model
- in the directory /EEG_identity_disguising/validation
- train the ResNet classifier
    - Select the ResNet model (ResNet18|ResNet34|ResNet50) and the classification task (alcoholism|stimulus|id) when training, for example:
```
python3 resnet_classification_model.py --model ResNet18 --feature alcoholism
```
*The script above train a ResNet-18 model to perform the alcoholism detection task.*
### Test the EEG disguising Model
- several essential option in the script
    - `--name` checkpoints folder name
    - `--classifier` which classifier is used for evaluation
    - `--test_all` use this argument to require the validation on the model from all checkpoints saved every 5 training epochs
    - `--which_epoch` if `test_all` not specified, select one epoch of the model to test  
- the example script below load checkpoints from the checkpoints `cycada_alcoholism_v2`, and requires to test the model from all training epochs   
```
python validation.py --name cycada_alcoholism_v2 --resize_or_crop=None --loadSize=32 --fineSize=32 --which_model_netD n_layers --n_layers_D 3 --model test --no_flip --batchSize 32 --dataset_mode EEGsingle  --dataroot ./datasets/eeg/ --data uci_eeg_images_validation_within.mat  --which_direction BtoA --phase train --how_many 100000 --classifier ResNet34 --test_all
```
- the scripts below are the ones with the experiemnt results I put in the report.
```
python validation.py --name cycada_alcoholism_v2 --resize_or_crop=None --loadSize=32 --fineSize=32 --which_model_netD n_layers --n_layers_D 3 --model test --no_flip --batchSize 32 --dataset_mode EEGsingle  --dataroot ./datasets/eeg/ --data uci_eeg_images_validation_within.mat  --which_direction BtoA --phase train --how_many 100000 --classifier ResNet34 --which_epoch 190
```
```
python validation.py --name cycada_stimulus_v2 --resize_or_crop=None --loadSize=32 --fineSize=32 --which_model_netD n_layers --n_layers_D 3 --model test --no_flip --batchSize 32 --dataset_mode EEGsingle  --dataroot ./datasets/eeg/ --data uci_eeg_images_validation_within.mat  --which_direction BtoA --phase train --how_many 100000 --classifier ResNet34 --which_epoch 45
```
```
python validation.py --name cycada_combined_v2 --resize_or_crop=None --loadSize=32 --fineSize=32 --which_model_netD n_layers --n_layers_D 3 --model test --no_flip --batchSize 32 --dataset_mode EEGsingle  --dataroot ./datasets/eeg/ --data uci_eeg_images_validation_within.mat  --which_direction BtoA --phase train --how_many 100000 --classifier ResNet34 --which_epoch 150
```
```
python validation.py --name cyclegan_v2 --resize_or_crop=None --loadSize=32 --fineSize=32 --which_model_netD n_layers --n_layers_D 3 --model test --no_flip --batchSize 32 --dataset_mode EEGsingle  --dataroot ./datasets/eeg/ --data uci_eeg_images_validation_within.mat  --which_direction BtoA --phase train --how_many 100000 --classifier ResNet34 --which_epoch 40
```
The full version of the project with code, datasets and checkpoints are uploaded to this share link: [code, datasets and checkpoints](https://anu365-my.sharepoint.com/:f:/g/personal/u6783346_anu_edu_au/EgceXDJhJvhBuzYdsF0ELogBhISm7VaMaH-rBRqMHj_DPQ?e=tjOhO2)

**You may need to downgrade scipy to 1.1.0*
```
pip install scipy==1.1.0
```
if you find this code useful, please kindly cite 

```
@article{yao2020information,
  title={Information-preserving Feature Filter for Short-term EEG signals},
  author={Yao, Yue and Plested, Josephine and Gedeon, Tom},
  journal={Neurocomputing},
  year={2020},
  publisher={Elsevier}
}
```