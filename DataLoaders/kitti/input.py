import os
import torch
import glob
from torch.utils.data import Dataset, DataLoader
import numpy as np
import cv2
from PIL import Image
from torchvision import transforms
from augment import *
import matplotlib.pyplot as plt
from DataLoaders.kitti.data import KITTI_raw
import time
import copy

KITTI_EXCLUDES = os.path.join('DataLoaders', 'kitti', 'kitti_excludes')

class KITTI_Dataset(Dataset):

    def __init__(self, datadir=None, crop_size=None, min_scale=1.0, max_scale=1.0,
                 noise_stddev=0.0, min_contrast=0.0, max_contrast=0.0,
                 brightness_stddev=0.0, min_colour=1.0, max_colour=1.0,
                 max_translation_x=0.0, max_translation_y=0.0, max_rotation=0.0,
                 min_gamma=1.0, max_gamma=1.0, create_border_mask=None, num_frames=2,
                 return_swap_sequences=False):

        super(KITTI_Dataset, self).__init__()

        if crop_size is None:
            self.crop_size = (256, 256)
            print('Argument crop_size must be specified to manually set the '
                  'resolution of images during training or else all the images will be cropped to a size of (256, 256')
        else:
            assert isinstance(crop_size, (int, tuple))  # crop_size must be int or tuple
            if isinstance(crop_size, int):
                self.crop_size = (crop_size, crop_size)
            else:
                self.crop_size = crop_size

        if create_border_mask is not None:
            self.border_mask = self.create_border_mask(self.crop_size, ratio=0.1)
        else:
            self.border_mask = None

        self.RandomCrop = RandomSizedCrop(self.crop_size)
        self.RandomAffine = RandomAffine(self.crop_size, min_scale=min_scale, max_scale=max_scale,
                                         max_translation_x=max_translation_x,
                                         max_translation_y=max_translation_y,
                                         max_rotation=max_rotation)
        self.RandomPhotometric = RandomPhotometric(noise_stddev=noise_stddev,
                                                   min_contrast=min_contrast,
                                                   max_contrast=max_contrast,
                                                   brightness_stddev=brightness_stddev,
                                                   min_colour=min_colour, max_colour=max_colour,
                                                   min_gamma=min_gamma, max_gamma=max_gamma,
                                                   num_images=num_frames)

        self.RandomHorizontalFlip = RandomHorizontalFlip()
        self.ToTensor = ToTensor()

        self.datadir = datadir
        self.num_frames = num_frames

        dataset = KITTI_raw(datadir=self.datadir,
                            excludes_files=KITTI_EXCLUDES, max_distance=10)

        self.image_paths_dict = dataset.image_path_dict
        self.normalization = dataset.channel_mean

        self.img_sequences = []

        for key, value in self.image_paths_dict.items():
            seq = [value[i:i+num_frames] for i in np.arange(0, len(value) - num_frames + 1)]

            if return_swap_sequences is True:
                swap_seq = [k[::-1] for k in seq]
                seq += swap_seq

            self.img_sequences += seq

        # self.img_sequences = [[os.path.join('first.png'), os.path.join('second.png')]]

        # self.img_sequences = self.img_sequences[0:10]

    def __len__(self):
        return len(self.img_sequences)

    def __getitem__(self, idx):

        img_seq = self.img_sequences[idx]

        img_list = [np.array(Image.open(img_seq[i])) for i in range(self.num_frames)]

        sample = np.expand_dims(np.concatenate(img_list, 2), 0)

        sample = self.ToTensor(sample)
        sample = self.RandomCrop(sample)

        border_mask = copy.deepcopy(self.border_mask)

        if self.border_mask is not None:
            sample = torch.cat([sample, border_mask], 1)

        sample = self.RandomAffine(sample)
        sample = self.RandomHorizontalFlip(sample)

        sample_new = {}

        if self.border_mask is not None:
            border_mask = sample[:, -1, :, :]
            sample_new['border_mask'] = border_mask
            sample = sample[:, :-1, :, :]

        sample = self.RandomPhotometric(sample.view(self.num_frames, 3,
                                                                  self.crop_size[0], self.crop_size[1]))

        sample = torch.split(sample, 1, 0)

        for i in range(self.num_frames):
            sample_new['image' + str(i+1)] = torch.squeeze(sample[i], 0)

        return sample_new

    @staticmethod
    def create_border_mask(size, ratio=0.1):

        min_dim = size[0] if size[0] <= size[1] else size[1]
        border = int(np.ceil(ratio * min_dim))

        mask = np.ones((size[0] - 2 * border, size[1] - 2 * border))
        mask = np.pad(mask, ((border, border), (border, border)))

        mask = torch.tensor(mask, dtype=torch.float32)

        return mask.reshape(1, 1, size[0], size[1])

