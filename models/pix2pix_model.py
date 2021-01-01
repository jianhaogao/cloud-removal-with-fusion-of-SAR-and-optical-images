import torch
from util.image_pool import ImagePool
from .base_model import BaseModel
from . import networks
import random
import torch.nn as nn
from util.util import tensor2im
import torchvision.transforms as transforms
from torchvision import models
import copy
import math
def create_loss_model(vgg, end_layer, use_maxpool=True):
    vgg = copy.deepcopy(vgg)
    model = nn.Sequential()
    i = 0
    for layer in list(vgg):

        if i > end_layer:
            break

        if isinstance(layer, nn.Conv2d):
            name = "conv_" + str(i)
            model.add_module(name, layer)

        if isinstance(layer, nn.ReLU):
            name = "relu_" + str(i)
            model.add_module(name, layer)

        if isinstance(layer, nn.MaxPool2d):
            name = "pool_" + str(i)
            if use_maxpool:
                model.add_module(name, layer)
            else:
                avgpool = nn.AvgPool2d(kernel_size=layer.kernel_size, stride=layer.stride, padding=layer.padding)
                model.add_module(name, avgpool)
        i += 1
    return model


class Pix2PixModel(BaseModel):
    def name(self):
        return 'Pix2PixModel'

    @staticmethod
    def modify_commandline_options(parser, is_train=True):

        # changing the default values to match the pix2pix paper
        # (https://phillipi.github.io/pix2pix/)
        parser.set_defaults(pool_size=0, no_lsgan=True, norm='batch')
        parser.set_defaults(dataset_mode='unaligned')
        parser.set_defaults(netG='resnet_9blocks')
        return parser

    def initialize(self, opt):
        BaseModel.initialize(self, opt)
        self.isTrain = opt.isTrain
        # specify the training losses you want to print out. The program will call base_model.get_current_losses
        self.loss_names = ['G1_L1', 'G2_L1','Res_L1','Perc','G_GAN']
        # specify the images you want to save/display. The program will call base_model.get_current_visuals
        self.visual_names = ['Simu', 'Fusion', 'mask','SAR','OPT','Corrupt']
        # specify the models you want to save to the disk. The program will call base_model.save_networks and base_model.load_networks
        if self.isTrain:
            self.model_names = ['G1','G2', 'D']
        else:  # during test time, only load Gs
            self.model_names = ['G1', 'G2']
        # load/define networks
        self.netG1 = networks.define_G(opt.input_nc, opt.output_nc, opt.ngf, opt.netG, opt.norm,
                                      not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids)
        self.netG2 = networks.define_G(9, opt.output_nc, opt.ngf, opt.netG, opt.norm,
                                      not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids)

        if self.isTrain:
            use_sigmoid = opt.no_lsgan
            self.netD = networks.define_D(opt.input_nc + opt.output_nc, opt.ndf, opt.netD,
                                          opt.n_layers_D, opt.norm, use_sigmoid, opt.init_type, opt.init_gain, self.gpu_ids)


        if self.isTrain:

            self.criterionGAN = networks.GANLoss(use_lsgan=not opt.no_lsgan).to(self.device)
            self.criterionL1 = torch.nn.L1Loss()
            self.resLoss = torch.nn.L1Loss()
            self.Loss_P = torch.nn.MSELoss()


            vgg16 = models.vgg16(pretrained=True).features
            vgg16=vgg16.to(self.device)
            self.vgg_loss = create_loss_model(vgg16, 8)
            for param in self.vgg_loss.parameters():
                param.requires_grad = False

            self.optimizers = []
            self.optimizer_G1 = torch.optim.Adam(self.netG1.parameters(),
                                                lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_G2 = torch.optim.Adam(self.netG2.parameters(),
                                                lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_D = torch.optim.Adam(self.netD.parameters(),
                                                lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizers.append(self.optimizer_G1)
            self.optimizers.append(self.optimizer_G2)
            self.optimizers.append(self.optimizer_D)

    def set_input(self, input):

        self.SAR = input['SAR'].to(self.device)
        self.OPT = input['OPT'].to(self.device)
        self.Corrupt = input['Corrupt'].to(self.device)
        self.mask = input['mask'].to(self.device)
        self.image_paths = input['SAR_paths']

    def forward(self):
        self.mask=torch.abs(1-self.mask)
        self.Simu = self.netG1(self.SAR)
        self.mean = torch.randn([1,3,1,1])
        self.mean[:,0,:,:] = torch.mean(self.Simu[:,0,:,:])
        self.mean[:,1,:,:] = torch.mean(self.Simu[:,1,:,:])
        self.mean[:,2,:,:] = torch.mean(self.Simu[:,2,:,:])
        self.std = torch.randn([1,3,1,1])
        self.std[:,0,:,:] = torch.std(self.Simu[:,0,:,:])
        self.std[:,1,:,:] = torch.std(self.Simu[:,1,:,:])
        self.std[:,2,:,:] = torch.std(self.Simu[:,2,:,:])

        self.meanrand = torch.randn([1,3,1,1])
        self.stdrand = torch.randn([1,3,1,1])

        self.mean = self.mean.cuda()
        self.std = self.std.cuda()
        self.meanrand = self.meanrand.cuda()
        self.stdrand = self.stdrand.cuda()
        index = [0,1,2]
        random.shuffle(index)

        self.Simurand = (self.Simu - self.mean)/self.std * (self.std+self.stdrand) + self.meanrand
        self.Simurand = self.Simurand[:,index,:,:]
        #通道交换
        
        self.inter1 = torch.cat((self.SAR,self.Corrupt, self.Simurand.detach()), 1)
        self.Fusion = self.netG2(self.inter1)

        self.Res_OPT = self.OPT * self.mask
        self.Res_Fusion = self.Fusion * self.mask

        #self.vgg_OPT = self.vgg_loss(self.OPT)
        #self.vgg_Fusion = self.vgg_loss(self.Fusion)

    def backward_D(self):
        # Fake
        # stop backprop to the generator by detaching fake_B
        fake_SO = torch.cat((self.SAR, self.Fusion), 1)
        pred_fake = self.netD(fake_SO.detach())
        self.loss_D_fake = self.criterionGAN(pred_fake, False)

        # Real
        real_SO = torch.cat((self.SAR, self.OPT), 1)
        pred_real = self.netD(real_SO)
        self.loss_D_real = self.criterionGAN(pred_real, True)

        # Combined loss
        self.loss_D = (self.loss_D_fake + self.loss_D_real) * 0.5

        self.loss_D.backward()

    def backward_G1(self):
        self.loss_G1_L1 = self.criterionL1(self.Simu, self.OPT) 

        self.loss_G1 = self.loss_G1_L1

        self.loss_G1.backward()

    def backward_G2(self):
        fake_SO = torch.cat((self.SAR, self.Fusion), 1)
        pred_fake = self.netD(fake_SO)
        self.loss_G_GAN = self.criterionGAN(pred_fake, True)*0.0001


        

        self.vggOPT=self.vgg_loss(self.OPT)
        self.vggFusion=self.vgg_loss(self.Fusion)


        self.loss_G2_L1 = self.criterionL1(self.Fusion, self.OPT) * 10
        self.loss_Res_L1=self.resLoss(self.Res_OPT, self.Res_Fusion)*10
        self.loss_Perc=self.Loss_P(self.vggOPT,self.vggFusion)* 0.8


        self.loss_G = self.loss_G_GAN + self.loss_G2_L1 + self.loss_Perc + self.loss_Res_L1

        self.loss_G.backward()

    def optimize_parameters(self):
        self.forward()

        self.optimizer_G1.zero_grad()
        self.backward_G1()
        self.optimizer_G1.step()


        self.set_requires_grad(self.netD, False)
        self.optimizer_G2.zero_grad()
        self.backward_G2()
        self.optimizer_G2.step()

        self.set_requires_grad(self.netD, True)
        self.optimizer_D.zero_grad()
        self.backward_D()
        self.optimizer_D.step()
