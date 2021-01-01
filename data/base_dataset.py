import torch.utils.data as data
from PIL import Image
import torchvision.transforms as transforms


class BaseDataset(data.Dataset):
    def __init__(self):
        super(BaseDataset, self).__init__()

    def name(self):
        return 'BaseDataset'

    @staticmethod
    def modify_commandline_options(parser, is_train):
        return parser

    def initialize(self, opt):
        pass

    def __len__(self):
        return 0


def get_transform1(opt):
    transform_list = []
    transform_list += [transforms.ToTensor(),
                       transforms.Normalize((0.5, 0.5, 0.5),
                                            (0.5, 0.5, 0.5))]
    return transforms.Compose(transform_list)

def get_transform2(opt):
    transform_list = []
    transform_list += [transforms.ToTensor()]
    return transforms.Compose(transform_list)


