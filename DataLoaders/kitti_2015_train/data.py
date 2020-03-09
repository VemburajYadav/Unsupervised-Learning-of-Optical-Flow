import os
import numpy as np
import glob
import random
import copy

class KITTI_2015_train(object):

    VALIDATION_FRAMES = np.array([ 25, 171,  42,  96, 157, 185,  83,  77,  79,  23,  12, 145, 123,
       190, 110, 183,  88,  86, 119,  35, 102, 130,  87,  31, 155,  89,
       141,  41, 175, 173, 121, 146,  74, 132,  70,  39,  53, 105,  75,
        51,  21,  58, 194, 179,  27,  73,  62,  78, 108,  24])

    def __init__(self, datadir=None, mode='all'):

        self.datadir = datadir

        image_2_dir = os.path.join(self.datadir, 'image_2')
        image_3_dir = os.path.join(self.datadir, 'image_3')
        flow_noc_dir = os.path.join(self.datadir, 'flow_noc')
        flow_occ_dir = os.path.join(self.datadir, 'flow_occ')

        image_2_pngs = sorted(glob.glob(os.path.join(image_2_dir + '/*')))
        image_3_pngs = sorted(glob.glob(os.path.join(image_3_dir + '/*')))
        flow_noc_pngs = sorted(glob.glob(os.path.join(flow_noc_dir + '/*')))
        flow_occ_pngs = sorted(glob.glob(os.path.join(flow_occ_dir + '/*')))

        num_pairs = len(flow_noc_pngs)

        image_2_pairs = np.array([[image_2_pngs[i], image_2_pngs[i+1]] for i in range(0, num_pairs*2-1, 2)])
        image_3_pairs = np.array([[image_3_pngs[i], image_3_pngs[i+1]] for i in range(0, num_pairs*2-1, 2)])

        self.data_dict = {}

        if mode == 'all':
            train_image_2_pairs = copy.deepcopy(image_2_pairs)
            train_image_3_pairs = copy.deepcopy(image_3_pairs)
            train_flow_noc = copy.deepcopy(flow_noc_pngs)
            train_flow_occ = copy.deepcopy(flow_occ_pngs)

            zip_all = list(zip(train_image_2_pairs, train_image_3_pairs, train_flow_noc, train_flow_occ))

            random.shuffle(zip_all)

            train_image_2_pairs, train_image_3_pairs, train_flow_noc, train_flow_occ \
                = zip(*zip_all)

            self.data_dict['image_2_pairs'] = train_image_2_pairs
            self.data_dict['image_3_pairs'] = train_image_3_pairs
            self.data_dict['flow_noc'] = train_flow_noc
            self.data_dict['flow_occ'] = train_flow_occ

        elif mode == 'val':
            val_image_2_pairs = list(image_2_pairs[self.VALIDATION_FRAMES])
            val_image_3_pairs = list(image_3_pairs[self.VALIDATION_FRAMES])
            flow_noc_pngs = np.array(flow_noc_pngs)
            val_flow_noc = list(flow_noc_pngs[self.VALIDATION_FRAMES])

            flow_occ_pngs = np.array(flow_occ_pngs)
            val_flow_occ = list(flow_occ_pngs[self.VALIDATION_FRAMES])

            self.data_dict['image_2_pairs'] = val_image_2_pairs
            self.data_dict['image_3_pairs'] = val_image_3_pairs
            self.data_dict['flow_noc'] = val_flow_noc
            self.data_dict['flow_occ'] = val_flow_occ

        elif mode == 'train':
            train_image_2_pairs = list(np.delete(image_2_pairs, self.VALIDATION_FRAMES, axis=0))
            train_image_3_pairs = list(np.delete(image_3_pairs, self.VALIDATION_FRAMES, axis=0))

            flow_noc_pngs = np.array(flow_noc_pngs)
            train_flow_noc = list(np.delete(flow_noc_pngs, self.VALIDATION_FRAMES))

            flow_occ_pngs = np.array(flow_occ_pngs)
            train_flow_occ = list(np.delete(flow_occ_pngs, self.VALIDATION_FRAMES))

            zip_all = list(zip(train_image_2_pairs, train_image_3_pairs, train_flow_noc, train_flow_occ))

            random.shuffle(zip_all)

            train_image_2_pairs, train_image_3_pairs, train_flow_noc, train_flow_occ \
                = zip(*zip_all)

            self.data_dict['image_2_pairs'] = train_image_2_pairs
            self.data_dict['image_3_pairs'] = train_image_3_pairs
            self.data_dict['flow_noc'] = train_flow_noc
            self.data_dict['flow_occ'] = train_flow_occ











