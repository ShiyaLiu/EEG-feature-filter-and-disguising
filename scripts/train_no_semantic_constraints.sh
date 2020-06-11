python train.py --name cyclegan --resize_or_crop=None --loadSize=32 --fineSize=32 --which_model_netD n_layers --n_layers_D 3 --model cycle_gan --lambda_A 1 --lambda_B 1 --lambda_identity 0 --no_flip --batchSize 64 --dataset_mode EEG --dataroot ./datasets/eeg/ --data_real uci_eeg_images_train_within_extra.mat --data_dummy eeg_dummy_images_w_label_step3_within_extra.mat --which_direction BtoA


