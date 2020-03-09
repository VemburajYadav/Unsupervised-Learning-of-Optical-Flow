import configparser
import os
import numpy as np

def config_dict(config_path='./config.ini'):
    """Returns the config as dictionary,
    where the elements have intuitively correct types.
    """

    config = configparser.ConfigParser()
    config.read(config_path)

    d = dict()
    for section_key in config.sections():
        sd = dict()
        section = config[section_key]
        for key in section:
            val = section[key]
            try:
                sd[key] = int(val)
            except ValueError:
                try:
                    sd[key] = float(val)
                except ValueError:
                    try:
                        sd[key] = section.getboolean(key)
                    except ValueError:
                        sd[key] = val
        d[section_key] = sd
    return d


def convert_input_strings(config_dct, dirs):

    if 'manual_decay_iters' in config_dct and 'manual_decay_lrs' in config_dct:
        iters_lst = config_dct['manual_decay_iters'].split(',')
        lrs_lst = config_dct['manual_decay_lrs'].split(',')
        iters_lst = [int(i) for i in iters_lst]
        lrs_lst = [float(l) for l in lrs_lst]
        config_dct['manual_decay_iters'] = iters_lst
        config_dct['manual_decay_lrs'] = lrs_lst
        config_dct['num_iters'] = sum(iters_lst)

    if 'normalization' in config_dct:
        normalization = config_dct['normalization'].split(',')
        normalization = [float(l) for l in normalization]
    else:
        normalization = [0., 0., 0.]

    config_dct['normalization'] = normalization

    if config_dct.get('finetune'):
        finetune = []
        for name in config_dct['finetune'].split(","):
            experiment_dir = os.path.join(dirs['log'], name)
            final_ckpt_dir = os.path.join(experiment_dir, 'Final_CKPTs')
            intermediate_ckpt_dir = os.path.join(experiment_dir, 'Intermediate_CKPTs')

            if not os.path.isdir(final_ckpt_dir):
                raise NotADirectoryError('The directory {} to load checkpoints from the experiment {}'
                                         'does not exist'.format(final_ckpt_dir, name))

            ckpt_file = get_latest_checkpoint(final_ckpt_dir)

            if ckpt_file is None:
                raise FileNotFoundError(' The directory {} containing final checkpoints of the experiment {}'
                                        'is empty. This could be due to the experiment {} being interrupted '
                                        'during training. Try copying the desired checkpoint file from the '
                                        'directory {} to the directory {} and execute again'.format(final_ckpt_dir,
                                        name, name, intermediate_ckpt_dir, final_ckpt_dir))

            finetune.append(os.path.join(final_ckpt_dir, ckpt_file))

        config_dct['finetune'] = finetune

    return config_dct


def get_latest_checkpoint(ckpt_dir):
    """ Get the latest checkpoint file form a directory"""

    listfiles = os.listdir(ckpt_dir)

    if len(listfiles) == 0:
        return None
    else:
        file_split = listfiles[0].split('_')
        extension = file_split[-1].split('.')[-1]

        basename = ''
        for i in range(len(file_split) - 1):
            basename = basename + file_split[i] + '_'

        listfiles_step = [int(file.split('_')[-1].split('.')[0]) for file in listfiles]
        listfiles_step = np.array(listfiles_step)
        global_step = listfiles_step.max()

        return basename + str(global_step) + '.' + extension


