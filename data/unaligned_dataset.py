import os.path
from data.base_dataset import BaseDataset, get_transform1,get_transform2
from data.image_folder import make_dataset
from PIL import Image
import random
import torchvision.transforms as transforms

class UnalignedDataset(BaseDataset):
    @staticmethod
    def modify_commandline_options(parser, is_train):
        return parser

    def initialize(self, opt):
        self.opt = opt
        self.root = opt.dataroot
        self.dir_SAR = os.path.join(opt.dataroot, opt.phase, 'SAR')
        self.dir_OPT = os.path.join(opt.dataroot, opt.phase, 'OPT')
        self.dir_Corrupt = os.path.join(opt.dataroot, opt.phase, 'Corrupt')
        self.dir_mask = os.path.join(opt.dataroot, opt.phase, 'mask')

        self.SAR_paths = make_dataset(self.dir_SAR)
        self.OPT_paths = make_dataset(self.dir_OPT)
        self.Corrupt_paths = make_dataset(self.dir_Corrupt)
        self.mask_paths = make_dataset(self.dir_mask)

        self.SAR_paths = sorted(self.SAR_paths)
        self.OPT_paths = sorted(self.OPT_paths)
        self.Corrupt_paths = sorted(self.Corrupt_paths)
        self.mask_paths = sorted(self.mask_paths)

        self.SAR_size = len(self.SAR_paths)
        self.OPT_size = len(self.OPT_paths)
        self.Corrupt_size = len(self.Corrupt_paths)
        self.mask_size = len(self.mask_paths)

        self.transform1 = get_transform1(opt)
        self.transform2 = get_transform2(opt)

    def __getitem__(self, index):
        SAR_path = self.SAR_paths[index % self.SAR_size]
        OPT_path = self.OPT_paths[index % self.OPT_size]
        Corrupt_path = self.Corrupt_paths[index % self.Corrupt_size]
        mask_path = self.mask_paths[index % self.mask_size]



        SAR_img = Image.open(SAR_path).convert('RGB')
        OPT_img = Image.open(OPT_path).convert('RGB')
        Corrupt_img = Image.open(Corrupt_path).convert('RGB')
        mask_img = Image.open(mask_path).convert('RGB')


        SAR = self.transform1(SAR_img)
        OPT = self.transform1(OPT_img)
        Corrupt = self.transform1(Corrupt_img)
        mask = self.transform2(mask_img)

        return {'SAR': SAR, 'OPT': OPT, 'Corrupt': Corrupt, 'mask': mask,
                'SAR_paths': SAR_path}

    def __len__(self):
        return self.SAR_size

    def name(self):
        return 'UnalignedDataset'
