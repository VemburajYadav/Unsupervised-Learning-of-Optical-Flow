import os
import torch
import glob
from torch.utils.data import Dataset, DataLoader
import numpy as np
from PIL import Image
from torchvision import transforms
from augment import *
import matplotlib.pyplot as plt
from DataLoaders.kitti_2012_multi_view.data import KITTI_2012_MultiView
from DataLoaders.kitti_2015_multi_view.data import KITTI_2015_MultiView
import time
import copy
import png


class KITTI_2012_2015_MultiView_Dataset(Dataset):

    def __init__(self, datadirs=None, crop_size=None, min_scale=1.0, max_scale=1.0,
                 noise_stddev=0.0, min_contrast=0.0, max_contrast=0.0,
                 brightness_stddev=0.0, min_colour=1.0, max_colour=1.0,
                 max_translation_x=0.0, max_translation_y=0.0, max_rotation=0.0,
                 min_gamma=1.0, max_gamma=1.0, num_frames=2, random_horizontal_flip=False,
                 max_distance=1, return_swap_sequences=False, create_border_mask=None):

        super(KITTI_2012_2015_MultiView_Dataset, self).__init__()

        if crop_size is None:
            # self.crop_size = (256, 256)
            # print('Argument crop_size must be specified to manually set the '
            #       'resolution of images during training or else all the images will be cropped to a size of (256, 256')
            self.crop_size = crop_size
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

        dataset2012 = KITTI_2012_MultiView(datadir=datadirs[0])
        dataset2015 = KITTI_2015_MultiView(datadir=datadirs[1])

        self.data_dict_2012 = dataset2012.scene_imgs
        self.data_dict_2015 = dataset2015.scene_imgs

        self.data_dict = {}

        self.data_dict['image_2'] = self.data_dict_2012['image_2'] + self.data_dict_2015['image_2']
        self.data_dict['image_3'] = self.data_dict_2012['image_3'] + self.data_dict_2015['image_3']

        min_frame_num = 10 - max_distance
        max_frame_num = 12 + max_distance

        self.num_frames = num_frames

        self.img_seq_list = []

        for key, value in self.data_dict.items():
            for scene in value:
                self.img_seq_list.append(scene[0:min_frame_num])
                self.img_seq_list.append(scene[max_frame_num:])

        self.img_frames = []

        for seq in self.img_seq_list:
            self.img_frames += [seq[i:i+num_frames] for i in range(0, len(seq) - num_frames + 1)]

        if crop_size:
            self.RandomCrop = RandomSizedCrop(self.crop_size)

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

        self.normalization = [0., 0., 0.]

    def __len__(self):
        return len(self.img_frames)

    def __getitem__(self, idx):

        img_seq = self.img_frames[idx]

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
            sample = self.RandomPhotometric(sample[:, :-1, :, :].view(self.num_frames, 3,
                                                                      self.crop_size[0], self.crop_size[1]))

        else:
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


# datadir = ['../../../KITTI_Optical_flow/data_stereo_flow_multiview/training',
#            '../../../KITTI_Optical_flow/data_scene_flow_multiview/training']
#
# dataset = KITTI_2012_2015_MultiView_Dataset(datadirs=datadir, num_frames=5, crop_size=(320, 1152),
#                                             min_scale=0.9, max_scale=1.1,
#                                             noise_stddev=0.04, min_contrast=-0.3, max_contrast=0.3,
#                                             brightness_stddev=0.02, min_colour=0.9, max_colour=1.1,
#                                             max_translation_x=0.0, max_translation_y=0.0, max_rotation=10.0,
#                                             min_gamma=0.7, max_gamma=1.3, create_border_mask=True)
#
# from torch.utils.data import DataLoader
# ds = DataLoader(dataset, shuffle=True, batch_size=4, drop_last=True)
#
# dataiter = iter(ds)
# sample = next(dataiter)
#
# plt.figure()
# plt.subplot(3,2,1)
# plt.imshow(sample['image1'].permute(0, 2, 3, 1)[0].numpy() / 255)
#
# plt.subplot(3,2,2)
# plt.imshow(sample['image2'].permute(0, 2, 3, 1)[0].numpy() / 255)
#
# plt.subplot(3,2,3)
# plt.imshow(sample['image3'].permute(0, 2, 3, 1)[0].numpy() / 255)
#
# plt.subplot(3,2,4)
# plt.imshow(sample['image4'].permute(0, 2, 3, 1)[0].numpy() / 255)
#
# plt.subplot(3,2,5)
# plt.imshow(sample['image5'].permute(0, 2, 3, 1)[0].numpy() / 255)
#
# plt.subplot(3,2,6)
# plt.imshow(sample['border_mask'].permute(0, 2, 3, 1)[0].view(320, 1152).numpy())
#
# plt.show()
#
# plt.figure()
# plt.subplot(3,2,1)
# plt.imshow(sample['image1'].permute(0, 2, 3, 1)[3].numpy() / 255)
#
# plt.subplot(3,2,2)
# plt.imshow(sample['image2'].permute(0, 2, 3, 1)[3].numpy() / 255)
#
# plt.subplot(3,2,3)
# plt.imshow(sample['image3'].permute(0, 2, 3, 1)[3].numpy() / 255)
#
# plt.subplot(3,2,4)
# plt.imshow(sample['image4'].permute(0, 2, 3, 1)[3].numpy() / 255)
#
# plt.subplot(3,2,5)
# plt.imshow(sample['image5'].permute(0, 2, 3, 1)[3].numpy() / 255)
#
# plt.subplot(3,2,6)
# plt.imshow(sample['border_mask'].permute(0, 2, 3, 1)[3].view(320, 1152).numpy())
#
# plt.show()
#
#