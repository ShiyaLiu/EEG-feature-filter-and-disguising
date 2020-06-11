from .base_model import BaseModel
from . import networks

'''
test mode
modified from the code 
https://github.com/jhoffman/pytorch-CycleGAN-and-pix2pix/blob/e484612d83449d05024a7d5fd2e012be53faad85/models/test_model.py
'''
class TestModel(BaseModel):
    def name(self):
        return 'TestModel'

    def initialize(self, opt):
        assert(not opt.isTrain)
        BaseModel.initialize(self, opt)

        # specify the training losses you want to print out. The program will call base_model.get_current_losses
        self.loss_names = []
        # specify the images you want to save/display. The program will call base_model.get_current_visuals
        self.visual_names = ['real_A', 'fake_B']
        # specify the models you want to save to the disk. The program will call base_model.save_networks and base_model.load_networks
        self.model_names = ['G_A']

        self.netG_A = networks.define_G(opt.input_nc, opt.output_nc,
                                      opt.ngf, opt.which_model_netG,
                                      opt.norm, not opt.no_dropout,
                                      opt.init_type,
                                      self.gpu_ids)

    def set_input(self, input):
        # we need to use single_dataset mode
        self.real_A = input['O'].to(self.device)
        #self.image_paths = input['A_paths']
        self.label_A_alcoholism = input['O_label_alcoholism'].to(self.device)
        self.label_A_stimulus = input['O_label_stimulus'].to(self.device)
        self.label_A_id = input['O_label_id'].to(self.device)

    def forward(self):
        self.fake_B = self.netG_A(self.real_A)
