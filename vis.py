import os
from options.test_options import TestOptions
from data import CreateDataLoader
from models import create_model
from util.visualizer import save_images
from util import html
import numpy as np
import torch

import matplotlib.pyplot as plt 

device = 'cuda' if torch.cuda.is_available() else 'cpu'

'''
Create samples of orginal and disguised EEG images
'''

def visulize(img, output):
    img_in = img[0,:,:,:].cpu().numpy()
    img_in -= np.min(img_in)
    img_in /= np.max(img_in)

    plt.clf()
    plt.subplot(1,1,1)
    plt.axis('off')
    plt.imshow(img_in)
    plt.savefig(output, bbox_inches='tight')

if __name__ == '__main__':
    opt = TestOptions().parse()
    opt.nThreads = 1   # test code only supports nThreads = 1
    opt.batchSize = 64   # test code only supports batchSize = 1
    opt.serial_batches = True  # no shuffle
    opt.no_flip = True  # no flip
    opt.display_id = -1  # no visdom display
    
    #load data
    print('create data loader')
    data_loader = CreateDataLoader(opt)
    print('load data..')
    dataset = data_loader.load_data()
    print('end loading')
    
    #load model
    model = create_model(opt)
    model.setup(opt)
  
    if not os.path.isdir('imgs'):
        os.makedirs('imgs')
            
    for i, data in enumerate(dataset):

        model.set_input(data)
        model.test()

        ori_img = model.real_A.permute(0,3,2,1)
        fake_img = model.fake_B.permute(0,3,2,1)           
        #print(ori_img.shape)
        
        #set output path
        path_ori = 'imgs/originalEEG_%d.png'%i
        path_fake = 'imgs/disguisedEEG_%d.png'%i
        #visualize and save images
        visulize(ori_img, path_ori)
        visulize(fake_img, path_fake)

        break

