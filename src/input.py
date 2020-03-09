import os
import torch
import glob
from torch.utils.data import Dataset, DataLoader
import numpy as np
from PIL import Image
from skimage import io
import cv2
from torchvision import transforms
from augment import *
import matplotlib.pyplot as plt
import time

class Dataset(Dataset):

    def __init__(self, datadirs=None, crop_size=None, min_scale=1.0, max_scale=1.0,
               noise_stddev=0.0, min_contrast=0.0, max_contrast=0.0,
               brightness_stddev=0.0, min_colour=1.0, max_colour=1.0,
               max_translation_x=0.0, max_translation_y=0.0, max_rotation=0.0,
               min_gamma=1.0, max_gamma=1.0, create_border_mask=None):

        super(MyDataset, self).__init__()

        if crop_size is None:
            self.crop_size = (256, 256)
            print('Argument crop_size must be specified to manually set the '
                                 'resolution of images during training or else all the images will be cropped to a size of (256, 256')
        else:
            assert isinstance(crop_size, (int, tuple)) # crop_size must be int or tuple
            if isinstance(crop_size, int):
                self.crop_size = (crop_size, crop_size)
            else:
                self.crop_size = crop_size

        if create_border_mask is not None:
            self.border_mask = (self.create_border_mask(self.crop_size, ratio=0.1) * 255).astype(np.uint8)
        else:
            self.border_mask = None

        self.RandomCrop = RandomSizedCrop(self.crop_size)
        self.RandomAffine = RandomAffine(min_scale=min_scale, max_scale=max_scale, max_translation_x=max_translation_x,
                         max_translation_y=max_translation_y, max_rotation=max_rotation)
        self.RandomPhotometric = RandomPhotometric(noise_stddev=noise_stddev, min_contrast=min_contrast, max_contrast=max_contrast,
                              brightness_stddev=brightness_stddev, min_colour=min_colour, max_colour=max_colour,
                              min_gamma=min_gamma, max_gamma=max_gamma)
        self.RandomHorizontalFlip = RandomHorizontalFlip()

        if type(datadirs) is not list:
            self.datadirs = [datadirs]
        else:
            self.datadirs = datadirs

        self.listfiles = [sorted(glob.glob(os.path.join(self.datadirs[i] + '/*')))
                          for i in range(len(self.datadirs))]

        self.img_pairs = []

        for j in range(len(self.datadirs)):
            self.img_pairs += [[self.listfiles[j][i], self.listfiles[j][i+1]]
                          for i in np.arange(0, len(self.listfiles[j]) - 1, 2)]

    def __len__(self):
        return len(self.img_pairs)

    def __getitem__(self, idx):

        img_pair = self.img_pairs[idx]

        img1 = cv2.imread(img_pair[0])
        img2 = cv2.imread(img_pair[1])

        sample = {'image1' : img1, 'image2' : img2}

        sample = self.RandomCrop(sample)

        if self.border_mask is not None:
            sample = {'image1': sample['image1'], 'image2': sample['image2'], 'border_mask': self.border_mask}

        sample = self.RandomAffine(sample)

        if self.border_mask is not None:
            self.border_mask = sample['border_mask']
            sample = {'image1' : sample['image1'], 'image2' : sample['image2']}

        sample = self.RandomPhotometric(sample)
        sample = self.RandomHorizontalFlip(sample)

        sample_new = {}

        for key, value in sample.items():
            sample_new[key] = cv2.cvtColor(sample[key], cv2.COLOR_BGR2RGB)

        if self.border_mask is not None:
            sample_new['border_mask'] = self.border_mask / 255

        return sample_new

    def create_border_mask(self, size, ratio=0.1):

        min_dim = size[0] if size[0] <= size[1] else size[1]
        border = int(np.ceil(ratio * min_dim))

        mask = np.ones((size[0] - 2 * border, size[1] - 2 * border))
        mask = np.pad(mask, ((border, border), (border, border)))

        return mask


datapath1 = '/media/vemburaj/9d072277-d226-41f6-a38d-1db833dca2bd/KITTI_VIdeos/2011_09_26_drive_0005_sync/' \
           '2011_09_26/2011_09_26_drive_0005_sync/image_02/data'

datapath2 = '/media/vemburaj/9d072277-d226-41f6-a38d-1db833dca2bd/KITTI_VIdeos/2011_09_26_drive_0002_sync/' \
           '2011_09_26/2011_09_26_drive_0002_sync/image_02/data'

datapath = [datapath1, datapath2]

ds = MyDataset(datadirs=datapath, crop_size=(320, 1152), min_scale=0.9, max_scale=1.1,
               noise_stddev=0.04, min_contrast=-0.3, max_contrast=0.3,
               brightness_stddev=0.02, min_colour=0.9, max_colour=1.1,
               min_gamma=0.7, max_gamma=1.5, create_border_mask=True)


batch_size = 3
dl = DataLoader(ds, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=8)

dataiter = iter(dl)

start = time.time()
for i in range(10):
    batch = next(dataiter)
end = time.time() - start

plt.figure()

for i in range(2 * batch_size):
    plt.subplot(batch_size, 2, i+1)

    if i % 2 == 0:
        plt.imshow(batch['image1'][i//2, :, :])
    else:
        plt.imshow(batch['image2'][i//2, :, :])

plt.show()