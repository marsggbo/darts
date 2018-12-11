#coding=utf-8
import os
import pickle
import time

import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder

class Cutout(object):
    def __init__(self, length):
        self.length = length

    def __call__(self, img):
        h, w = img.size(1), img.size(2)
        mask = np.ones((h, w), np.float32)
        y = np.random.randint(h)
        x = np.random.randint(w)

        y1 = np.clip(y - self.length // 2, 0, h)
        y2 = np.clip(y + self.length // 2, 0, h)
        x1 = np.clip(x - self.length // 2, 0, w)
        x2 = np.clip(x + self.length // 2, 0, w)

        mask[y1: y2, x1: x2] = 0.
        mask = torch.from_numpy(mask)
        mask = mask.expand_as(img)
        img *= mask
        return img

def get_dataloaders(args):
    image_size = args.image_size
    batch_size = args.batch_size
    data_type = args.data_type
    path = args.data_path

    data_transforms = {
        'train': transforms.Compose([
            transforms.Resize(image_size),
            transforms.CenterCrop(image_size),
            transforms.RandomVerticalFlip(),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            # transforms.ColorJitter(0.5, 0.5, 0.5, 0.2),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'valid': transforms.Compose([
            transforms.Resize(image_size),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'test': transforms.Compose([
            transforms.Resize(image_size),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    }
    if args.cutout:
        data_transforms['train'].transforms.append(Cutout(args.cutout_length))
    

    if data_type == 0:
        # ImageFolder
        data_folders = ['train', 'valid', 'test']
        datas = {}
        data_size = {}
        data_loaders = {}

        for df in data_folders:
            print('Generating {} DataLoader...'.format(df))
            data_path = os.path.join(path, df)
            datas[df] = ImageFolder(data_path, data_transforms[df])
            data_size[df] = len(datas[df])
            data_loaders[df] = DataLoader(
                datas[df], batch_size=batch_size, shuffle=True, num_workers=4)
        return data_loaders['train'], data_loaders['valid'], data_loaders['test']
    
def test():
    '''
    test the code
    '''
    class params():
        def __init__(self):
            self.image_size = 32 
            self.batch_size = 16
            self.data_type = 0
            self.data_path = "../../enas/data/skin5"
            self.cutout=False

    args = params()
    loaders = get_dataloaders(args)
    print(loaders)

if __name__ == '__main__':
    test()    