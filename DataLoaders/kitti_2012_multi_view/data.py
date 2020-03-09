import os
import numpy as np
import glob
import random


class KITTI_2012_MultiView(object):

    def __init__(self, datadir=None):

        self.datadir = datadir

        image_2_dir = os.path.join(self.datadir, 'image_2')
        image_3_dir = os.path.join(self.datadir, 'image_3')

        self.imgs2 = sorted(glob.glob(os.path.join(image_2_dir + '/*')))
        self.imgs3 = sorted(glob.glob(os.path.join(image_3_dir + '/*')))

        print(len(self.imgs2))
        num = 0

        self.imgs_2 = []
        self.imgs_3 = []

        remove_scenes = ['000031', '000082', '000114']
        for i in range(len(self.imgs2)):
            img = self.imgs2[i]
            img_num = img.split('/')[-1].split('.png')[0].split('_')[0]
            if img_num not in remove_scenes:
                self.imgs_2.append(img)

        for i in range(len(self.imgs3)):
            img = self.imgs3[i]
            img_num = img.split('/')[-1].split('.png')[0].split('_')[0]
            if img_num not in remove_scenes:
                self.imgs_3.append(img)

        self.scene_imgs = {'image_2': [self.imgs_2[i:i+21] for i in range(0, len(self.imgs_2), 21)],
                           'image_3': [self.imgs_3[i:i+21] for i in range(0, len(self.imgs_3), 21 )]}

# datadir = '../../../KITTI_Optical_flow/data_scene_flow_multiview/training'
#
# dataset = KITTI_2012_MultiView(datadir=datadir)