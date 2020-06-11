import os
from options.test_options import TestOptions
from data import CreateDataLoader
from models import create_model
from util.visualizer import save_images
from util import html
import numpy as np
import torch
from sklearn.metrics import recall_score

from validation import autoencoder_classification_alcoholism
from validation import Image_wise_autoencoders
from validation import autoencoder_classification_stimulus
from validation import autoencoder_classification_personal_identity

from resnet import ResNet18
from resnet import ResNet34
from resnet import ResNet50

import torch.backends.cudnn as cudnn


#from util.tsne_visual import embd_visual, embd_visual_group


import matplotlib.pyplot as plt 
#from plotly.graph_objs import Scatter, Layout

#plotly.offline.init_notebook_mode(connected=True)
device = 'cuda' if torch.cuda.is_available() else 'cpu'


if __name__ == '__main__':
    opt = TestOptions().parse()
    opt.nThreads = 1   # test code only supports nThreads = 1
    opt.batchSize = 64   # test code only supports batchSize = 1
    opt.serial_batches = True  # no shuffle
    opt.no_flip = True  # no flip
    opt.display_id = -1  # no visdom display
    print('create data loader')
    data_loader = CreateDataLoader(opt)
    print('load data..')
    dataset = data_loader.load_data()
    print('end loading')
    model = create_model(opt)
    model.setup(opt)

    how_many = len(data_loader)
    num_classes_alc = 2
    num_classes_stimulus = 5
    num_classes_id = 122
    # create website
    #web_dir = os.path.join(opt.results_dir, opt.name, '%s_%s' % (opt.phase, opt.which_epoch))
    #webpage = html.HTML(web_dir, 'Experiment = %s, Phase = %s, Epoch = %s' % (opt.name, opt.phase, opt.which_epoch))
    # test

    if opt.classifier == 'AE':
        autoencoder_alcoholism = Image_wise_autoencoders.CNN().cuda()
        autoencoder_stimulus = Image_wise_autoencoders.CNN().cuda()
        autoencoder_id = Image_wise_autoencoders.CNN().cuda()
    
        classifier_alcoholism = autoencoder_classification_alcoholism.ClassificationNet().cuda()
        classifier_stimulus = autoencoder_classification_stimulus.ClassificationNet().cuda()
        classifier_id = autoencoder_classification_personal_identity.ClassificationNet().cuda()
        
        autoencoder_alcoholism.load_state_dict(torch.load(
            'validation/autoencoder_checkpoints/Image-wise_autoencoders_disease_within_augmented_3.pkl'))
        autoencoder_stimulus.load_state_dict(torch.load('validation/autoencoder_checkpoints/Image-wise_autoencoders_within_stimulus.pkl'))
        autoencoder_id.load_state_dict(torch.load('validation/autoencoder_checkpoints/Image-wise_autoencoders_within_id.pkl'))
        classifier_alcoholism.load_state_dict(torch.load(
            'validation/autoencoder_checkpoints/final_classification_model_disease_within_augmented_3.pkl'))
        classifier_stimulus.load_state_dict(torch.load('validation/autoencoder_checkpoints/final_classification_model-stimulus.pkl'))
        classifier_id.load_state_dict(torch.load('validation/autoencoder_checkpoints/final_classification_model-id.pkl'))
        
        autoencoder_alcoholism.eval()
        autoencoder_stimulus.eval()
        autoencoder_id.eval()  
   
    else:
        if opt.classifier == 'ResNet18':
            classifier_alcoholism = ResNet18(num_classes_alc)
            classifier_stimulus = ResNet18(num_classes_stimulus)
            classifier_id = ResNet18(num_classes_id)
        elif opt.classifier == 'ResNet34':
            classifier_alcoholism = ResNet34(num_classes_alc)
            classifier_stimulus = ResNet34(num_classes_stimulus)
            classifier_id = ResNet34(num_classes_id)
        elif opt.classifier == 'ResNet50':
            classifier_alcoholism = ResNet50(num_classes_alc)
            classifier_stimulus = ResNet50(num_classes_stimulus)
            classifier_id = ResNet50(num_classes_id)
            
        classifier_alcoholism = classifier_alcoholism.to(device)
        classifier_stimulus = classifier_stimulus.to(device)
        classifier_id = classifier_id.to(device)
        
        if device == 'cuda':
            classifier_alcoholism = torch.nn.DataParallel(classifier_alcoholism)
            classifier_stimulus = torch.nn.DataParallel(classifier_stimulus)
            classifier_id = torch.nn.DataParallel(classifier_id)            
            cudnn.benchmark = True 
            
        checkpoint_alcoholism = torch.load('./validation/resnet_checkpoints/alcoholism/%s/ckpt.pth'%opt.classifier)
        classifier_alcoholism.load_state_dict(checkpoint_alcoholism['net'])
        checkpoint_stimulus = torch.load('./validation/resnet_checkpoints/stimulus/%s/ckpt.pth'%opt.classifier)
        classifier_stimulus.load_state_dict(checkpoint_stimulus['net'])
        checkpoint_id = torch.load('./validation/resnet_checkpoints/id/%s/ckpt.pth'%opt.classifier)
        classifier_id.load_state_dict(checkpoint_id['net'])
     
    classifier_alcoholism.eval()
    classifier_stimulus.eval
    classifier_id.eval()

    disease_acc = []
    stimulus_acc = []
    
    alcoholism_preds_real = []
    alcoholism_preds_disguised = []    
    alcoholism_targets = []
    

    test_fake_alc_correct = 0
    test_real_alc_correct = 0

    test_real_stimulus_correct = 0
    test_fake_stimulus_correct = 0

    test_real_id_correct = 0
    test_fake_id_correct = 0   


    for i, data in enumerate(dataset):
        model.set_input(data)
        model.test()

        real_eeg = model.real_A
        disguised_eeg = model.fake_B
        
        ###### alcoholism ######
        ## real EEG
        target_alcoholism = model.label_A_alcoholism
        alcoholism_targets.append(target_alcoholism)
        if opt.classifier == 'AE':
            real_alcoholism_feature, _ = autoencoder_alcoholism(real_eeg)
            output = classifier_alcoholism (real_alcoholism_feature)
        else:
            output = classifier_alcoholism (real_eeg)
        
        _, predict = output.max(1)
        alcoholism_preds_real.append(predict)

        test_real_alc_correct += np.sum((predict == target_alcoholism.long()).data.cpu().numpy())   

        ## disguised EEG
        if opt.classifier == 'AE':
            fake_alcoholism_feature, _ = autoencoder_alcoholism(disguised_eeg)       
            output = classifier_alcoholism(fake_alcoholism_feature)
        else:
            output = classifier_alcoholism(disguised_eeg)
        
        _, predict = output.max(1)
        alcoholism_preds_disguised.append(predict)
        test_fake_alc_correct += np.sum((predict == target_alcoholism.long()).data.cpu().numpy())

        print('disease label')
        print(target_alcoholism)
        print('disease fake')
        print (predict)
        
        ###### stimulus ######
        target_stimulus = model.label_A_stimulus
        ## real EEG
        if opt.classifier == 'AE':
            real_stimulus_feature, _ = autoencoder_stimulus(real_eeg)
            output_stimulus = classifier_stimulus(real_stimulus_feature)
        else:
            output_stimulus = classifier_stimulus(real_eeg)

        _, predict = torch.max(output_stimulus, 1)

        test_real_stimulus_correct += np.sum((predict == target_stimulus.long()).data.cpu().numpy())
        
        ## disguised EEG
        if opt.classifier == 'AE':
            fake_stimulus_feature, _ = autoencoder_stimulus(disguised_eeg)
            output_stimulus = classifier_stimulus(fake_stimulus_feature)
        else:
            output_stimulus = classifier_stimulus(disguised_eeg)
        _, predict = torch.max(output_stimulus, 1)
        test_fake_stimulus_correct += np.sum((predict == target_stimulus.long()).data.cpu().numpy())
        
        ###### ID ######
        target_id = model.label_A_id
        ## real EEG
        if opt.classifier == 'AE':
            real_id_feature, _ = autoencoder_id(real_eeg)
            output_id = classifier_id(real_id_feature)
        else:
            output_id = classifier_id(real_eeg)
        _, predict = torch.max(output_id, 1)
        test_real_id_correct += np.sum((predict == target_id.long()).data.cpu().numpy())
        
        ## disguised EEG
        if opt.classifier == 'AE':
            fake_id_feature, _ = autoencoder_id(disguised_eeg)
            output_id = classifier_id(fake_id_feature)
        else:
            output_id = classifier_id(disguised_eeg)
        _, predict = torch.max(output_id, 1)
        test_fake_id_correct += np.sum((predict == target_id.long()).data.cpu().numpy())
    
    #### report ####
    alcoholism_targets = torch.cat(alcoholism_targets).cpu()    
    alcoholism_preds_real = torch.cat(alcoholism_preds_real).cpu()
    alcoholism_preds_disguised = torch.cat(alcoholism_preds_disguised).cpu()
    print (how_many)
    print ("original alcoholism accuracy is:", test_real_alc_correct/ how_many) 
    print ("final alcoholism accuracy is:", test_fake_alc_correct/ how_many) 
    print()
    print("alcoholism_sensibility_real:", recall_score(alcoholism_targets, alcoholism_preds_real, pos_label=1))
    print("alcoholism_specificity_disguised:", recall_score(alcoholism_targets, alcoholism_preds_disguised, pos_label=1))
    print()
    print("alcoholism_specificity_real:", recall_score(alcoholism_targets, alcoholism_preds_real, pos_label=0))
    print("alcoholism_specificity_disguised:", recall_score(alcoholism_targets, alcoholism_preds_disguised, pos_label=0))
    print()
    print ("original stimulus accuracy is:", test_real_stimulus_correct/ how_many) 
    print ("final stimulus accuracy is:", test_fake_stimulus_correct/ how_many)
    print()
    print ("original id accuracy is:", test_real_id_correct/ how_many) 
    print ("final id accuracy is:", test_fake_id_correct/ how_many) 
    