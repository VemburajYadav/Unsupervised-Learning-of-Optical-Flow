import os
import numpy as np
import glob
import random


class KITTI_2015_MultiView(object):

    def __init__(self, datadir=None):

        self.datadir = datadir

        image_2_dir = os.path.join(self.datadir, 'image_2')
        image_3_dir = os.path.join(self.datadir, 'image_3')

        self.imgs_2 = sorted(glob.glob(os.path.join(image_2_dir + '/*')))
        self.imgs_3 = sorted(glob.glob(os.path.join(image_3_dir + '/*')))

        self.scene_imgs = {'image_2': [self.imgs_2[i:i+21] for i in range(0, len(self.imgs_2), 21)],
                           'image_3': [self.imgs_3[i:i+21] for i in range(0, len(self.imgs_3), 21 )]}

datadir = '../../../KITTI_Optical_flow/data_scene_flow_multiview/training'

dataset = KITTI_2015_MultiView(datadir=datadir)