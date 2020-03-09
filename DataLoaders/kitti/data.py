import os
import glob
import numpy as np


class KITTI_raw(object):

    KITTI_DATA_RECORDING_DAYS = ['2011_09_29',
                                 '2011_09_28',
                                 '2011_09_30',
                                 '2011_09_26',
                                 '2011_10_03']

    def __init__(self, datadir=None, excludes_files=None, max_distance=None):

        assert datadir is not None  # path to the KITTI data directory must be specified

        self.datadir = datadir
        self.data_path_days = [os.path.join(self.datadir, path)
                               for path in sorted(self.KITTI_DATA_RECORDING_DAYS)]

        self.image_path_dict = {}
        self.image_dirs = []

        for i in range(len(self.data_path_days)):
            data_path_day = self.data_path_days[i]
            dirs = os.listdir(data_path_day)

            dirs = [os.path.join(data_path_day, j) for j in dirs if 'sync' in j]

            image_2_dirs = [os.path.join(j, 'image_02/data') for j in dirs]
            image_3_dirs = [os.path.join(j, 'image_03/data') for j in dirs]

            self.image_dirs = self.image_dirs + image_2_dirs + image_3_dirs

        for i in range(len(self.image_dirs)):
            self.image_path_dict[self.image_dirs[i]] = \
                sorted(glob.glob(os.path.join(self.image_dirs[i] + '/*')))

        if excludes_files is not None:
            if not os.path.isdir(os.path.join(excludes_files)):
                raise NotADirectoryError('Path specified as {} by exclude_files '
                                        'argument does not exist'.format(excludes_files))

            files = glob.glob(os.path.join(excludes_files + '/*'))

            indices_to_remove = {}

            for key, _ in self.image_path_dict.items():
                indices_to_remove[key] = np.array([])

            for file in files:
                with open(file, 'r') as f:
                    for line in f:
                        line = line.rstrip('\n')
                        splits = line.split(' ')

                        if splits[0].endswith('_10'):
                            path = splits[1].split('\\')
                            img_name, day, seq = path[-1], path[1], path[2]

                            path_image2 = os.path.join(self.datadir, day, seq, 'image_02/data')
                            path_image3 = os.path.join(self.datadir, day, seq, 'image_03/data')

                            frame_index = self.frame_name_to_num(img_name)

                            remove_indices = np.arange(frame_index - max_distance,
                                                       frame_index + max_distance + 1)

                            indices_to_remove[path_image2] = np.hstack([indices_to_remove[path_image2],
                                                                        remove_indices])
                            indices_to_remove[path_image3] = np.hstack([indices_to_remove[path_image3],
                                                                        remove_indices])

            for key, value in self.image_path_dict.items():
                filelist = np.array(value)
                new_filelist = list(np.delete(filelist, indices_to_remove[key]))
                self.image_path_dict[key] = new_filelist

        self.channel_mean = [104.920005, 110.1753, 114.785955]


    def frame_name_to_num(self, name):
        stripped = name.split('.')[0].lstrip('0')
        if stripped == '':
            return 0
        return int(stripped)




