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
from DataLoaders.kitti_2015_train.data import KITTI_2015_train
from DataLoaders.kitti_2012_train.data import KITTI_2012_train
import time
import copy
import png


class KITTI_2012_2015_Train_Dataset(Dataset):

    def __init__(self, datadirs=None, multiview_datadirs=None, crop_size=None, min_scale=1.0, max_scale=1.0,
                 noise_stddev=0.0, min_contrast=0.0, max_contrast=0.0,
                 brightness_stddev=0.0, min_colour=1.0, max_colour=1.0,
                 max_translation_x=0.0, max_translation_y=0.0, max_rotation=0.0,
                 min_gamma=1.0, max_gamma=1.0, num_frames=2, random_horizontal_flip=False,
                 return_flow_occ=True, return_flow_noc=True, mode='all', create_border_mask=None):

        super(KITTI_2012_2015_Train_Dataset, self).__init__()

        self.return_flow_noc = return_flow_noc
        self.num_frames = num_frames

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

        dataset_2012 = KITTI_2012_train(datadir=datadirs[0], mode=mode)
        dataset_2015 = KITTI_2015_train(datadir=datadirs[1], mode=mode)

        if self.num_frames > 2:
            self.multiview_path_2012 = os.path.join(multiview_datadirs[0], 'image_0')
            self.multiview_path_2015 = os.path.join(multiview_datadirs[1], 'image_2')

        self.data_dict = {}
        data_dict_2012 = dataset_2012.data_dict
        data_dcit_2015 = dataset_2015.data_dict

        self.data_dict['image_pairs'] = data_dict_2012['image_0_pairs'] + data_dcit_2015['image_2_pairs']
        self.data_dict['flow_noc'] = data_dict_2012['flow_noc'] + data_dcit_2015['flow_noc']
        self.data_dict['flow_occ'] = data_dict_2012['flow_occ'] + data_dcit_2015['flow_occ']

        if crop_size:
            self.RandomCrop = RandomSizedCrop(self.crop_size)

        self.RandomPhotometric = RandomPhotometric(noise_stddev=noise_stddev,
                                                   min_contrast=min_contrast,
                                                   max_contrast=max_contrast,
                                                   brightness_stddev=brightness_stddev,
                                                   min_colour=min_colour, max_colour=max_colour,
                                                   min_gamma=min_gamma, max_gamma=max_gamma,
                                                   num_images=num_frames)

        self.RandomHorizontalFlip = RandomHorizontalFlip()
        self.ToTensor = ToTensor()

    def __len__(self):
        return len(self.data_dict['image_pairs'])

    def __getitem__(self, idx):

        concat = []

        if 'image_0' in self.data_dict['image_pairs'][idx][0]:

            if self.num_frames > 2:
                img_path = (self.data_dict['image_pairs'][idx][0]).split('/')[-1].split('_')[0]

                print(self.data_dict['image_pairs'][idx][0])
                for i in range(self.num_frames - 2):
                    previous_frame = os.path.join(self.multiview_path_2012, img_path + '_0' + str(9-(self.num_frames-3)+i) + '.png')
                    img = np.array(Image.open(previous_frame), dtype=np.float32)
                    img = np.tile(np.reshape(img, (img.shape[0], img.shape[1], 1)), (1, 1, 3))
                    concat.append(np.expand_dims(img, 0))

            img_1 = np.array(Image.open(self.data_dict['image_pairs'][idx][0]), dtype=np.float32)
            img_2 = np.array(Image.open(self.data_dict['image_pairs'][idx][1]), dtype=np.float32)

            img_1 = np.tile(np.reshape(img_1, (img_1.shape[0], img_1.shape[1], 1)), (1, 1, 3))
            img_2 = np.tile(np.reshape(img_2, (img_2.shape[0], img_2.shape[1], 1)), (1, 1, 3))

        elif 'image_2' in self.data_dict['image_pairs'][idx][0]:
            if self.num_frames > 2:
                img_path = (self.data_dict['image_pairs'][idx][0]).split('/')[-1].split('_')[0]

                print(self.data_dict['image_pairs'][idx][0])

                for i in range(self.num_frames - 2):
                    previous_frame = os.path.join(self.multiview_path_2015, img_path + '_0' + str(9-(self.num_frames-3)+i) + '.png')
                    img = np.array(Image.open(previous_frame), dtype=np.float32)
                    img = np.tile(np.reshape(img, (img.shape[0], img.shape[1], 3)), (1, 1, 1))
                    concat.append(np.expand_dims(img, 0))

            img_1 = np.array(Image.open(self.data_dict['image_pairs'][idx][0]), dtype=np.float32)
            img_2 = np.array(Image.open(self.data_dict['image_pairs'][idx][1]), dtype=np.float32)

            img_1 = np.tile(np.reshape(img_1, (img_1.shape[0], img_1.shape[1], 3)), (1, 1, 1))
            img_2 = np.tile(np.reshape(img_2, (img_2.shape[0], img_2.shape[1], 3)), (1, 1, 1))

        flow_occ = self.read_flow_png(self.data_dict['flow_occ'][idx])

        flow_occ[:, :, 0:2] = (flow_occ[:, :, 0:2] - 2**15) / 64

        concat = concat + [np.expand_dims(img_1, 0), np.expand_dims(img_2, 0),
                           np.expand_dims(flow_occ, 0)]

        if self.return_flow_noc:
            flow_noc = self.read_flow_png(self.data_dict['flow_noc'][idx])
            flow_noc[:, :, 0:2] = (flow_noc[:, :, 0:2] - 2 ** 15) / 64
            concat.append(np.expand_dims(flow_noc, 0))

        sample = np.concatenate(concat, 0)

        sample = self.ToTensor(sample)

        if self.crop_size:
            sample = self.RandomCrop(sample)

        split = torch.split(sample, 1, 0)

        if self.return_flow_noc:
            img_photo = self.RandomPhotometric(torch.cat((split[:-2]), dim=0))
        else:
            img_photo = self.RandomPhotometric(torch.cat((split[:-1]), dim=0))

        sample_new = {}

        imgs = torch.split(img_photo, 1, 0)

        for i in range(self.num_frames):
            sample_new['image' + str(i+1)] = torch.squeeze(imgs[i], dim=0)

        if self.return_flow_noc:
            sample_new['flow_occ'] = split[-2][0, 0:2, :, :]
            sample_new['mask_occ'] = torch.unsqueeze(split[-2][0, 2, :, :], dim=0).clamp(0.0, 1.0)

            sample_new['flow_noc'] = split[-1][0, 0:2, :, :]
            sample_new['mask_noc'] = torch.unsqueeze(split[-1][0, 2, :, :], dim=0).clamp(0.0, 1.0)
        else:
            sample_new['flow_occ'] = split[-1][0, 0:2, :, :]
            sample_new['mask_occ'] = torch.unsqueeze(split[-1][0, 2, :, :], dim=0).clamp(0.0, 1.0)

        if self.border_mask is not None:
            sample_new['border_mask'] = torch.squeeze(self.border_mask, 0)

        return sample_new

    def read_flow_png(self, filename):
        flow_object = png.Reader(filename=filename)
        flow_direct = flow_object.asDirect()
        flow_data = list(flow_direct[2])
        (w, h) = flow_direct[3]['size']
        flow = np.zeros((h, w, 3), dtype=np.float32)

        for i in range(len(flow_data)):
            flow[i, :, 0] = flow_data[i][0::3]
            flow[i, :, 1] = flow_data[i][1::3]
            flow[i, :, 2] = flow_data[i][2::3]

        return flow

    @staticmethod
    def create_border_mask(size, ratio=0.1):

        min_dim = size[0] if size[0] <= size[1] else size[1]
        border = int(np.ceil(ratio * min_dim))

        mask = np.ones((size[0] - 2 * border, size[1] - 2 * border))
        mask = np.pad(mask, ((border, border), (border, border)))

        mask = torch.tensor(mask, dtype=torch.float32)

        return mask.reshape(1, 1, size[0], size[1])


