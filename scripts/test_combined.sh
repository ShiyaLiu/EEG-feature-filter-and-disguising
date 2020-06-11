python validation.py --name cycada_combined_v2 --resize_or_crop=None --loadSize=32 --fineSize=32 --which_model_netD n_layers --n_layers_D 3 --model test --no_flip --batchSize 32 --dataset_mode EEGsingle  --dataroot ./datasets/eeg/ --data uci_eeg_images_validation_within.mat  --which_direction BtoA --phase train --how_many 100000 --classifier ResNet34 --which_epoch 150

