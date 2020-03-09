import os
from shutil import copyfile, rmtree
import numpy as np
import torch
from .utils import config_dict

class Experiment():

    def __init__(self, name, ckpt_filename='ckpt', config_path=None, overwrite=False):
        self.global_config = config_dict(config_path=config_path)
        self.log_dir = os.path.join(self.global_config['dirs']['log'], name)

        self.train_dir = os.path.join(self.log_dir, 'train')
        self.eval_dir = os.path.join(self.log_dir, 'eval')
        self.intermediate_ckpt_dir = os.path.join(self.log_dir, 'Intermediate_CKPTs')
        self.final_ckpt_dir = os.path.join(self.log_dir, 'Final_CKPTs')

        self.global_step = 0
        self.ckpt_filename = ckpt_filename

        if not os.path.isdir(self.log_dir):
            os.makedirs(self.log_dir)
            os.makedirs(self.intermediate_ckpt_dir)
            os.makedirs(self.final_ckpt_dir)
            os.makedirs(self.train_dir)
            os.makedirs(self.eval_dir)
        else:
            if overwrite:
                rmtree(self.log_dir)
                os.makedirs(self.log_dir)
                os.makedirs(self.intermediate_ckpt_dir)
                os.makedirs(self.final_ckpt_dir)
                os.makedirs(self.train_dir)
                os.makedirs(self.eval_dir)
            else:
                if os.path.isdir(self.intermediate_ckpt_dir):
                    global_step = self.get_global_step(self.intermediate_ckpt_dir)
                    if global_step is None:
                        raise RuntimeError('Failed to restore checkpoints from "{}".'
                                           'Use --overwrite=True to clear the directory.'
                                           .format(self.intermediate_ckpt_dir))
                    else:
                        self.global_step = global_step
                else:
                    os.makedirs(self.intermediate_ckpt_dir)

                if not os.path.isdir(self.train_dir):
                    os.makedirs(self.train_dir)
                if not os.path.isdir(self.eval_dir):
                    os.makedirs(self.eval_dir)
                if not os.path.isdir(self.final_ckpt_dir):
                    os.makedirs(self.final_ckpt_dir)

        self.resume_experiment = False
        if self.global_step > 0:
            self.resume_experiment = True


    def get_global_step(self, ckpt_dir):
        """ Get the global step of latest checkpoint"""

        listfiles = os.listdir(ckpt_dir)

        if len(listfiles) == 0:
            return None
        else:
            listfiles_step = [int(file.split('_')[-1].split('.')[0]) for file in listfiles]
            listfiles_step = np.array(listfiles_step)
            global_step = listfiles_step.max()

            return global_step

    def conclude(self):
        """ Move the last checkpoint from Intermediate_CKPTs directory to Final_CKPTs directory"""

        global_step = self.get_global_step(self.intermediate_ckpt_dir)

        ckpt_filename = self.ckpt_filename + '_' + str(global_step)
        copyfile(os.path.join(self.intermediate_ckpt_dir, ckpt_filename),
                 os.path.join(self.final_ckpt_dir, ckpt_filename))

        print('Checkpoint file {} at final step {] is copied from {} to {}'.
              format(ckpt_filename, global_step, self.intermediate_ckpt_dir, self.final_ckpt_dir))





