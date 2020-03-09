import argparse
import os
from src.experiment import Experiment
import copy
from src.utils import convert_input_strings
from torch.utils.data import DataLoader
from src.core.train import Trainer
import torch

parser = argparse.ArgumentParser()
parser.add_argument('--config_path', type=str, default='./config.ini', help='Path to the config file')
parser.add_argument('--ex', type=str, default='my_experiment_pwc', help='Name for the Experiment')
parser.add_argument('--ckpt_filename', type=str, default='ckpt', help='Basename for saving the checkpoint files')
parser.add_argument('--ow', type=bool, default=False,
                    help='Whether to overwrite the contents in the log directory corresponding '
                         'to this experiment if it already exists')

opt = parser.parse_args()

experiment = Experiment(opt.ex, ckpt_filename=opt.ckpt_filename,
                        config_path=opt.config_path, overwrite=opt.ow)


dirs = experiment.global_config['dirs']
train_datadir = dirs['data']
val_datadir = dirs['val_data_path']
val_multiview_datadir = dirs.get('val_multiview_datapath')

run_config = experiment.global_config['run']


train_dataset = run_config['dataset']
val_dataset = run_config['val_dataset']

if val_dataset == 'kitti_2012_train':

    from DataLoaders.kitti_2012_train.input import KITTI_2012_Train_Dataset

    dataset_val = KITTI_2012_Train_Dataset(datadir=val_datadir, multiview_datadir=val_multiview_datadir,
                                           min_scale=1.0, max_scale=1.0,
                                           noise_stddev=0.0, min_contrast=0.0, max_contrast=0.0,
                                           brightness_stddev=0.0, min_colour=1.0, max_colour=1.0,
                                           max_translation_x=0.0, max_translation_y=0.0, max_rotation=0.0,
                                           min_gamma=1.0, max_gamma=1.0,
                                           num_frames=2, mode='all')

    val_dataLoader = DataLoader(dataset_val, batch_size=1, shuffle=True,
                                drop_last=True, num_workers=run_config['num_input_threads'])

elif val_dataset == 'kitti_ft':

    val_datadir_split = val_datadir.split(',')
    val_datadirs = [val_datadir_split[0], val_datadir_split[1]]

    from DataLoaders.kitti_2012_2015_train.input import KITTI_2012_2015_Train_Dataset

    dataset_val = KITTI_2012_2015_Train_Dataset(datadirs=val_datadirs,
                                                min_scale=1.0, max_scale=1.0,
                                                noise_stddev=0.0, min_contrast=0.0, max_contrast=0.0,
                                                brightness_stddev=0.0, min_colour=1.0, max_colour=1.0,
                                                max_translation_x=0.0, max_translation_y=0.0, max_rotation=0.0,
                                                min_gamma=1.0, max_gamma=1.0,
                                                num_frames=2, mode='val')


elif val_dataset == 'kitti_2015_train':

    from DataLoaders.kitti_2015_train.input import KITTI_2015_Train_Dataset

    dataset_val = KITTI_2015_Train_Dataset(datadir=val_datadir,
                                           min_scale=1.0, max_scale=1.0,
                                           noise_stddev=0.0, min_contrast=0.0, max_contrast=0.0,
                                           brightness_stddev=0.0, min_colour=1.0, max_colour=1.0,
                                           max_translation_x=0.0, max_translation_y=0.0, max_rotation=0.0,
                                           min_gamma=1.0, max_gamma=1.0,
                                           num_frames=2, mode='val')

if train_dataset == 'kitti':
    cconfig = copy.deepcopy(experiment.global_config['train'])
    cconfig.update(experiment.global_config['train_kitti'])
    cconfig = convert_input_strings(cconfig, dirs)
    cconfig['global_step'] = experiment.global_step
    cconfig['resume_experiment'] = experiment.resume_experiment
    cconfig['supervised'] = False

    from DataLoaders.kitti.input import KITTI_Dataset

    dataset_train = KITTI_Dataset(datadir=train_datadir, crop_size=(cconfig['height'], cconfig['width']),
                                  min_scale=0.9, max_scale=1.1,
                                  noise_stddev=0.04, min_contrast=-0.3, max_contrast=0.3,
                                  brightness_stddev=0.02, min_colour=0.9, max_colour=1.1,
                                  max_translation_x=0.0, max_translation_y=0.0, max_rotation=0.0,
                                  min_gamma=0.7, max_gamma=1.5, create_border_mask=cconfig['border_mask'],
                                  num_frames=2,
                                  return_swap_sequences=cconfig['swap_sequences'])

    trainer = Trainer(dataset_train, dataset_val, batch_size=run_config['batch_size'],
                      params=cconfig, num_workers=run_config['num_input_threads'],
                      train_summaries_dir=experiment.train_dir,
                      eval_summaries_dir=experiment.eval_dir,
                      training_checkpoints_save_dir=experiment.intermediate_ckpt_dir,
                      final_checkpoints_save_dir=experiment.final_ckpt_dir,
                      experiment=experiment,
                      normalization=cconfig['normalization'],
                      device_ids=0)

    trainer.train()
    experiment.conclude()

elif train_dataset == 'kitti_ft':
    cconfig = copy.deepcopy(experiment.global_config['train'])
    cconfig.update(experiment.global_config['train_kitti_ft'])
    cconfig = convert_input_strings(cconfig, dirs)
    cconfig['global_step'] = experiment.global_step
    cconfig['resume_experiment'] = experiment.resume_experiment
    cconfig['supervised'] = True

    train_datadir_split = train_datadir.split(',')
    train_datadirs = [train_datadir_split[0], train_datadir_split[1]]

    from DataLoaders.kitti_2012_2015_train.input import KITTI_2012_2015_Train_Dataset

    dataset_train = KITTI_2012_2015_Train_Dataset(datadirs=train_datadirs, crop_size=(cconfig['height'], cconfig['width']),
                                                  min_scale=1.0, max_scale=1.0,
                                                  noise_stddev=0.04, min_contrast=-0.3, max_contrast=0.3,
                                                  brightness_stddev=0.02, min_colour=0.9, max_colour=1.1,
                                                  max_translation_x=0.0, max_translation_y=0.0, max_rotation=0.0,
                                                  min_gamma=0.7, max_gamma=1.5, create_border_mask=cconfig['border_mask'],
                                                  num_frames=2, mode='train')

    trainer = Trainer(dataset_train, dataset_val, batch_size=run_config['batch_size'],
                      params=cconfig, num_workers=run_config['num_input_threads'],
                      train_summaries_dir=experiment.train_dir,
                      eval_summaries_dir=experiment.eval_dir,
                      training_checkpoints_save_dir=experiment.intermediate_ckpt_dir,
                      final_checkpoints_save_dir=experiment.final_ckpt_dir,
                      experiment=experiment,
                      normalization=cconfig['normalization'],
                      device_ids=0)

    trainer.train()
    experiment.conclude()


elif train_dataset == 'kitti_2012_train':
    cconfig = copy.deepcopy(experiment.global_config['train'])
    cconfig.update(experiment.global_config['train_2012_train'])
    cconfig = convert_input_strings(cconfig, dirs)
    cconfig['global_step'] = experiment.global_step
    cconfig['resume_experiment'] = experiment.resume_experiment
    cconfig['supervised'] = True


    from DataLoaders.kitti_2012_train.input import KITTI_2012_Train_Dataset

    dataset_train = KITTI_2012_Train_Dataset(datadir=train_datadir, crop_size=(cconfig['height'], cconfig['width']),
                                             min_scale=1.0, max_scale=1.0,
                                             noise_stddev=0.04, min_contrast=-0.3, max_contrast=0.3,
                                             brightness_stddev=0.02, min_colour=0.9, max_colour=1.1,
                                             max_translation_x=0.0, max_translation_y=0.0, max_rotation=0.0,
                                             min_gamma=0.7, max_gamma=1.5, create_border_mask=cconfig['border_mask'],
                                             num_frames=2, mode='all')

    trainer = Trainer(dataset_train, dataset_val, batch_size=run_config['batch_size'],
                      params=cconfig, num_workers=run_config['num_input_threads'],
                      train_summaries_dir=experiment.train_dir,
                      eval_summaries_dir=experiment.eval_dir,
                      training_checkpoints_save_dir=experiment.intermediate_ckpt_dir,
                      final_checkpoints_save_dir=experiment.final_ckpt_dir,
                      experiment=experiment,
                      normalization=cconfig['normalization'],
                      device_ids=0)

    trainer.train()
    experiment.conclude()

elif train_dataset == 'kitti_2015_train':
    cconfig = copy.deepcopy(experiment.global_config['train'])
    cconfig.update(experiment.global_config['train_2015_train'])
    cconfig = convert_input_strings(cconfig, dirs)
    cconfig['global_step'] = experiment.global_step
    cconfig['resume_experiment'] = experiment.resume_experiment
    cconfig['supervised'] = True

    from DataLoaders.kitti_2015_train.input import KITTI_2015_Train_Dataset

    dataset_train = KITTI_2015_Train_Dataset(datadir=train_datadir, crop_size=(cconfig['height'], cconfig['width']),
                                             min_scale=1.0, max_scale=1.0,
                                             noise_stddev=0.04, min_contrast=-0.3, max_contrast=0.3,
                                             brightness_stddev=0.02, min_colour=0.9, max_colour=1.1,
                                             max_translation_x=0.0, max_translation_y=0.0, max_rotation=0.0,
                                             min_gamma=0.7, max_gamma=1.5, create_border_mask=cconfig['border_mask'],
                                             num_frames=2, mode='all')

    trainer = Trainer(dataset_train, dataset_val, batch_size=run_config['batch_size'],
                      params=cconfig, num_workers=run_config['num_input_threads'],
                      train_summaries_dir=experiment.train_dir,
                      eval_summaries_dir=experiment.eval_dir,
                      training_checkpoints_save_dir=experiment.intermediate_ckpt_dir,
                      final_checkpoints_save_dir=experiment.final_ckpt_dir,
                      experiment=experiment,
                      normalization=cconfig['normalization'],
                      device_ids=0)

    trainer.train()
    experiment.conclude()


elif train_dataset == 'kitti_2012_2015_multiview':
    cconfig = copy.deepcopy(experiment.global_config['train'])
    cconfig.update(experiment.global_config['train_kitti_2012_2015_multiview'])
    cconfig = convert_input_strings(cconfig, dirs)
    cconfig['global_step'] = experiment.global_step
    cconfig['resume_experiment'] = experiment.resume_experiment
    cconfig['supervised'] = False

    datadir_split = train_datadir.split(',')
    datadirs = [datadir_split[0], datadir_split[1]]
    from DataLoaders.kitti_2012_2015_multi_view.input import KITTI_2012_2015_MultiView_Dataset

    dataset_train = KITTI_2012_2015_MultiView_Dataset(datadirs=datadirs, crop_size=(cconfig['height'], cconfig['width']),
                                                      min_scale=0.9, max_scale=1.1,
                                                      noise_stddev=0.04, min_contrast=-0.3, max_contrast=0.3,
                                                      brightness_stddev=0.02, min_colour=0.9, max_colour=1.1,
                                                      max_translation_x=0.0, max_translation_y=0.0, max_rotation=0.0,
                                                      min_gamma=0.7, max_gamma=1.5, create_border_mask=cconfig['border_mask'],
                                                      num_frames=2,
                                                      return_swap_sequences=cconfig['swap_sequences'])

    trainer = Trainer(dataset_train, dataset_val, batch_size=run_config['batch_size'],
                      params=cconfig, num_workers=run_config['num_input_threads'],
                      train_summaries_dir=experiment.train_dir,
                      eval_summaries_dir=experiment.eval_dir,
                      training_checkpoints_save_dir=experiment.intermediate_ckpt_dir,
                      final_checkpoints_save_dir=experiment.final_ckpt_dir,
                      experiment=experiment,
                      normalization=cconfig['normalization'],
                      device_ids=0)

    trainer.train()
    experiment.conclude()
#
#
#
#
#
#
#
#

