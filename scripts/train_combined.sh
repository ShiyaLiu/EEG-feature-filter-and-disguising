python train.py --name cycada_combined --resize_or_crop=None --loadSize=32 --fineSize=32 --which_model_netD n_layers --n_layers_D 3 --model cycle_gan_semantic --lambda_A 1 --lambda_B 1 --lambda_identity 0 --no_flip --batchSize 64 --dataset_mode EEG --dataroot ./datasets/eeg/ --data_real uci_eeg_images_train_within.mat --data_dummy eeg_dummy_images_w_label_step3_within.mat --which_direction BtoA --feature combined --num_classes 10


