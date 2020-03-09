import os
import numpy as np
import glob
import random
import copy

class KITTI_2012_train(object):

    VALIDATION_FRAMES = np.array([128,  11, 182,  51,  97,  76, 127, 165, 166, 113, 183,  35,  12,
        18,  10, 136,  59, 119, 184, 171, 142, 126, 188,  58,  45, 153,
        89,  73, 170, 129, 125, 148,   1,  36, 187,   0,  95,  24, 134,
        15,  88, 120, 163,  27])

    def __init__(self, datadir=None, mode='all'):

        self.datadir = datadir

        image_0_dir = os.path.join(self.datadir, 'image_0')
        image_1_dir = os.path.join(self.datadir, 'image_1')
        colored_0_dir = os.path.join(self.datadir, 'colored_0')
        colored_1_dir = os.path.join(self.datadir, 'colored_1')
        flow_noc_dir = os.path.join(self.datadir, 'flow_noc')
        flow_occ_dir = os.path.join(self.datadir, 'flow_occ')

        image_0_pngs = sorted(glob.glob(os.path.join(image_0_dir + '/*')))
        image_1_pngs = sorted(glob.glob(os.path.join(image_1_dir + '/*')))
        colored_0_pngs = sorted(glob.glob(os.path.join(colored_0_dir + '/*')))
        colored_1_pngs = sorted(glob.glob(os.path.join(colored_1_dir + '/*')))
        flow_noc_pngs = sorted(glob.glob(os.path.join(flow_noc_dir + '/*')))
        flow_occ_pngs = sorted(glob.glob(os.path.join(flow_occ_dir + '/*')))

        num_pairs = len(flow_noc_pngs)

        image_0_pairs = np.array([[image_0_pngs[i], image_0_pngs[i+1]] for i in range(0, num_pairs*2-1, 2)])
        image_1_pairs = np.array([[image_1_pngs[i], image_1_pngs[i+1]] for i in range(0, num_pairs*2-1, 2)])
        colored_0_pairs = np.array([[colored_0_pngs[i], colored_0_pngs[i+1]] for i in range(0, num_pairs*2-1, 2)])
        colored_1_pairs = np.array([[colored_1_pngs[i], colored_1_pngs[i+1]] for i in range(0, num_pairs*2-1, 2)])

        self.data_dict = {}

        if mode == 'all':
            train_image_0_pairs = copy.deepcopy(image_0_pairs)
            train_image_1_pairs = copy.deepcopy(image_1_pairs)
            train_colored_0_pairs = copy.deepcopy(colored_0_pairs)
            train_colored_1_pairs = copy.deepcopy(colored_1_pairs)
            train_flow_noc = copy.deepcopy(flow_noc_pngs)
            train_flow_occ = copy.deepcopy(flow_occ_pngs)

            zip_all = list(zip(train_image_0_pairs, train_image_1_pairs, train_colored_0_pairs,
                               train_colored_1_pairs, train_flow_noc, train_flow_occ))

            random.shuffle(zip_all)

            train_image_0_pairs, train_image_1_pairs, train_colored_0_pairs, \
            train_colored_1_pairs, train_flow_noc, train_flow_occ \
                = zip(*zip_all)

            self.data_dict['image_0_pairs'] = train_image_0_pairs
            self.data_dict['image_1_pairs'] = train_image_1_pairs
            self.data_dict['colored_0_pairs'] = train_colored_0_pairs
            self.data_dict['colored_1_pairs'] = train_colored_1_pairs
            self.data_dict['flow_noc'] = train_flow_noc
            self.data_dict['flow_occ'] = train_flow_occ

        elif mode == 'val':
            val_image_0_pairs = list(image_0_pairs[self.VALIDATION_FRAMES])
            val_image_1_pairs = list(image_1_pairs[self.VALIDATION_FRAMES])
            val_colored_0_pairs = list(colored_0_pairs[self.VALIDATION_FRAMES])
            val_colored_1_pairs = list(colored_1_pairs[self.VALIDATION_FRAMES])

            flow_noc_pngs = np.array(flow_noc_pngs)
            val_flow_noc = list(flow_noc_pngs[self.VALIDATION_FRAMES])

            flow_occ_pngs = np.array(flow_occ_pngs)
            val_flow_occ = list(flow_occ_pngs[self.VALIDATION_FRAMES])

            self.data_dict['image_0_pairs'] = val_image_0_pairs
            self.data_dict['image_1_pairs'] = val_image_1_pairs
            self.data_dict['colored_0_pairs'] = val_colored_0_pairs
            self.data_dict['colored_1_pairs'] = val_colored_1_pairs
            self.data_dict['flow_noc'] = val_flow_noc
            self.data_dict['flow_occ'] = val_flow_occ

        elif mode == 'train':
            train_image_0_pairs = list(np.delete(image_0_pairs, self.VALIDATION_FRAMES, axis=0))
            train_image_1_pairs = list(np.delete(image_1_pairs, self.VALIDATION_FRAMES, axis=0))
            train_colored_0_pairs = list(np.delete(colored_0_pairs, self.VALIDATION_FRAMES, axis=0))
            train_colored_1_pairs = list(np.delete(colored_1_pairs, self.VALIDATION_FRAMES, axis=0))

            flow_noc_pngs = np.array(flow_noc_pngs)
            train_flow_noc = list(np.delete(flow_noc_pngs, self.VALIDATION_FRAMES))

            flow_occ_pngs = np.array(flow_occ_pngs)
            train_flow_occ = list(np.delete(flow_occ_pngs, self.VALIDATION_FRAMES))

            zip_all = list(zip(train_image_0_pairs, train_image_1_pairs, train_colored_0_pairs,
                               train_colored_1_pairs, train_flow_noc, train_flow_occ))

            random.shuffle(zip_all)

            train_image_0_pairs, train_image_1_pairs, train_colored_0_pairs, \
            train_colored_1_pairs, train_flow_noc, train_flow_occ \
                = zip(*zip_all)

            self.data_dict['image_0_pairs'] = train_image_0_pairs
            self.data_dict['image_1_pairs'] = train_image_1_pairs
            self.data_dict['colored_0_pairs'] = train_colored_0_pairs
            self.data_dict['colored_1_pairs'] = train_colored_1_pairs
            self.data_dict['flow_noc'] = train_flow_noc
            self.data_dict['flow_occ'] = train_flow_occ










