import torch
import torch.nn as nn
from src.core.flownet import Simple, Complex
import numpy as np
from src.utils import get_latest_checkpoint
import os
import copy
from src.core.unsupervised import unsupervised_loss_flownet, unsupervised_loss_pwcnet
from src.core.supervised import supervised_loss
from torch.optim import Adam, SGD
from src.core.flow_util import flow_error_avg, outlier_pct
from src.core.flownet import FLOW_SCALE
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from src.core.pwc_net import PWCNet


class Trainer(object):
    """ Class for training with validation """

    def __init__(self, train_dataset, eval_dataset, batch_size=4,
                 num_workers=1, params=None,
                 train_summaries_dir=None,
                 eval_summaries_dir=None,
                 training_checkpoints_save_dir=None,
                 final_checkpoints_save_dir=None,
                 experiment=None,
                 normalization=None,
                 device_ids=0):

        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.params = params
        self.train_summaries_dir = train_summaries_dir
        self.eval_summaries_dir = eval_summaries_dir
        self.training_checkpoints_save_dir = training_checkpoints_save_dir
        self.final_checkpoints_save_dir = final_checkpoints_save_dir
        self.experiment = experiment
        self.normalization = torch.tensor(normalization).view(1, len(normalization), 1, 1).to(device=device_ids)
        self.device_ids = device_ids

        self.train_dataLoader = DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True,
                                           drop_last=True, num_workers=self.num_workers, pin_memory=True)

        self.eval_dataLoader = DataLoader(self.eval_dataset, batch_size=1, shuffle=False,
                                          drop_last=True, num_workers=1, pin_memory=True)

        self.train_SummaryWriter = SummaryWriter(log_dir=self.train_summaries_dir)
        self.eval_SummaryWriter = SummaryWriter(log_dir=self.eval_summaries_dir)

        self.network = params.get('network')

        self.finetune = params.get('finetune')
        self.logs_interval = params.get('display_interval')
        self.create_train_steps_and_lr_decays()

        if not self.params['supervised']:
            self.get_loss_weights()

        if self.network == 'pwcnet':
            self.build_and_initialize_model_pwcnet()
            self.optimizer = Adam(params=self.pwcnet.parameters())

        elif self.network == 'flownet':
            self.flownet = params['flownet']
            self.build_and_initialize_model_flownet()

            if self.params.get('train_all'):
                self.optimizer = Adam(params=self.moduleFlownets.parameters())
            else:
                self.optimizer = Adam(params=self.moduleFlownets[-1].parameters())

        if self.params['resume_experiment']:
            self.restore_states_for_resume_experiment()

    def restore_states_for_resume_experiment(self):
        """ Checks if the training needs to be restarted from a specific checkpoint and restores the parameters """

        ckpt_file = os.path.join(self.training_checkpoints_save_dir,
                             get_latest_checkpoint(self.training_checkpoints_save_dir))
        ckpt_state = torch.load(ckpt_file)
        self.global_step = ckpt_state['global_step']
        if self.network == 'flownet':
            self.moduleFlownets.load_state_dict(ckpt_state['model_state_dict'])
        else:
            self.pwcnet.load_state_dict(ckpt_state['model_state_dict'])

        self.optimizer.load_state_dict(ckpt_state['optimizer_state_dict'])

        step_list = np.array(self.step_list)
        index = np.sum(step_list <= self.global_step)
        self.step_list = self.step_list[index:]
        self.lr_list = self.lr_list[index:]

    def train(self):

        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.manual_seed(2)
        torch.cuda.manual_seed(2)

        if self.params['supervised']:
            self.run_supervised()
        else:
            self.run_unsupervised()

        self.train_SummaryWriter.close()
        self.eval_SummaryWriter.close()

    def create_train_steps_and_lr_decays(self):
        """ Create intervals of training steps and learning rates"""

        decay_interval = self.params.get('decay_interval')
        save_interval = self.params.get('save_interval')
        decay_after = self.params.get('decay_after')
        num_iters = self.params.get('num_iters')
        manual_decay_iters = self.params.get('manual_decay_iters')
        manual_decay_lrs = self.params.get('manual_decay_lrs')
        learning_rate = self.params.get('learning_rate')

        self.step_list = []
        self.lr_list = []

        if not manual_decay_iters and not manual_decay_lrs:
            if not decay_after and not decay_interval:
                manual_decay_iters = [num_iters]
                manual_decay_lrs = [learning_rate]
            else:
                decay_after = decay_after if decay_after else 0
                manual_decay_iters = [decay_after]
                manual_decay_lrs = [learning_rate, learning_rate]
                num_decays = (num_iters - decay_after + decay_interval - 1) // decay_interval - 1
                manual_decay_iters += list(np.arange(decay_after, num_iters+1, decay_interval))[1:]
                manual_decay_lrs += [learning_rate / (2 ** i) for i in range(1, num_decays + 1, 1)]

        else:
            decay_iters_list = [manual_decay_iters[0]]
            decay_lrs_list = [manual_decay_lrs[0]]

            start = manual_decay_iters[0]
            for i in range(1, len(manual_decay_iters)):
                start = start + manual_decay_iters[i]
                decay_iters_list.append(start)
                decay_lrs_list.append(manual_decay_lrs[i])

            manual_decay_iters = copy.deepcopy(decay_iters_list)
            manual_decay_lrs = copy.deepcopy(decay_lrs_list)

        step_min = 0
        for i in range(len(manual_decay_iters)):
            step_max = manual_decay_iters[i]
            interval_iter_list = list(np.arange(step_min, step_max, save_interval)[1:])
            interval_iter_list.append(step_max)
            interval_lr_list = [manual_decay_lrs[i]] * len(interval_iter_list)

            self.step_list += interval_iter_list
            self.lr_list += interval_lr_list

            step_min = manual_decay_iters[i]

    def get_loss_weights(self):

        losses = ['occ_weight', 'fb_weight', 'grad_weight', 'photo_weight',
                  'ternary_weight', 'smooth_1st_weight', 'smooth_2nd_weight', 'smooth_2nd_edge_weight']

        param_keys = self.params.keys()

        self.loss_weights_dict = {}

        for loss in losses:
            if loss in param_keys:
                self.loss_weights_dict[loss] = self.params[loss]

    def build_and_initialize_model_flownet(self):
        """ Initialise the flownet model with random weights or weigths from other ecperiments (for finetuning) """

        moduleList = []

        for module in self.flownet:
            if module == 'C':
                moduleList.append(Complex())
            elif module == 'S':
                moduleList.append(Simple())
            else:
                raise ValueError('flownet {} has an invalid component. '
                                 'It must be either "C"s or "S"s'.format(self.flownet))

        self.moduleFlownets = nn.ModuleList(moduleList)

        if not self.params['resume_experiment']:
            self.global_step = 0

            if self.finetune:
                state_dict = self.moduleFlownets.state_dict()

                for i in range(len(self.finetune)):
                    weights = torch.load(self.finetune[i])['model_state_dict']

                    module_num = np.array([int(key.split('.')[0]) for key in weights.keys()]).max()

                    for key, value in weights.items():
                        splits = key.split('.')

                        if int(splits[0]) == module_num:
                            new_key = str(i)

                            for j in range(1, len(splits), 1):
                                new_key = new_key + '.' + splits[j]

                            # print(new_key)
                            state_dict[new_key] = value

                self.moduleFlownets.load_state_dict(state_dict)

        self.moduleFlownets[-1].train()

        if not self.params.get('train_all'):
            for i in range(len(self.moduleFlownets) - 1):
                self.moduleFlownets[i].requires_grad_(requires_grad=False).eval()
        else:
            for i in range(len(self.moduleFlownets) - 1):
                self.moduleFlownets[i].requires_grad_(requires_grad=True).train()

        self.moduleFlownets = self.moduleFlownets.to(device=self.device_ids)

    def build_and_initialize_model_pwcnet(self):
        """ Initialise the pwc-net model with random weights or weigths from other ecperiments (for finetuning) """

        self.pwcnet = PWCNet()

        if not self.params['resume_experiment']:
            self.global_step = 0

            if self.finetune:
                self.pwcnet.load_state_dict(torch.load(self.finetune[0])['model_state_dict'])
                state_dict = self.moduleFlownets.state_dict()

        self.pwcnet.train()
        self.pwcnet = self.pwcnet.to(device=self.device_ids)

    def run_unsupervised(self):
        """ Executes the unsupervised training loop"""

        dataiter = iter(self.train_dataLoader)

        for end_step, lr in zip(self.step_list, self.lr_list):

            for param_group in self.optimizer.param_groups:
                param_group['lr'] = lr

            while self.global_step <= end_step:
                try:
                    batch_input = next(dataiter)
                except StopIteration:
                    self.train_dataLoader = DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True,
                                                       drop_last=True, num_workers=self.num_workers, pin_memory=True)
                    dataiter = iter(self.train_dataLoader)
                    batch_input = next(dataiter)

                if self.network == 'flownet':
                    losses = unsupervised_loss_flownet(batch_input, module=self.moduleFlownets,
                                                       loss_weights_dict=self.loss_weights_dict,
                                                       params=self.params, return_flow=False,
                                                       normalization=self.normalization, device_ids=self.device_ids)
                else:
                    losses = unsupervised_loss_pwcnet(batch_input, module=self.pwcnet,
                                                       loss_weights_dict=self.loss_weights_dict,
                                                       params=self.params, return_flow=False,
                                                       normalization=self.normalization, device_ids=self.device_ids)


                self.optimizer.zero_grad()
                losses.backward()
                self.optimizer.step()

                if self.global_step % self.logs_interval == 0:
                    self.train_SummaryWriter.add_scalar('Train_Loss', losses, self.global_step)
                    print('step {}/{} ========================================= loss: {}'.
                          format(self.global_step, self.params.get('num_iters'), losses))

                self.global_step += 1

            if self.network == 'flownet':
                self.moduleFlownets.eval()
            else:
                self.pwcnet.eval()

            flow_error_noc, flow_error_occ, outlier_ratio_noc, outlier_ratio_occ = self.eval()

            print('step: {}, ========================================= val_flow_error_noc: {}, '
                  'val_flow_error_occ: {}, outlier_ratio_noc: {}, outlier_ratio_occ: {}'.
                  format(self.global_step, flow_error_noc, flow_error_occ, outlier_ratio_noc,
                         outlier_ratio_occ))

            if self.network == 'flownet':
                if self.params.get('train_all'):
                    self.moduleFlownets.requires_grad_(requires_grad=True).train()
                else:
                    self.moduleFlownets[-1].requires_grad_(requires_grad=True).train()

                save_state_dict = {
                    'model_state_dict': self.moduleFlownets.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'global_step': self.global_step,
                    'val_flow_error_occ': flow_error_occ,
                    'val_flow_error_noc': flow_error_noc,
                    'val_outlier_ratio_occ': outlier_ratio_occ,
                    'val_outlier_ratio_noc': outlier_ratio_noc
                }

            else:
                self.pwcnet.requires_grad_(requires_grad=True).train()

                save_state_dict = {
                    'model_state_dict': self.pwcnet.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'global_step': self.global_step,
                    'val_flow_error_occ': flow_error_occ,
                    'val_flow_error_noc': flow_error_noc,
                    'val_outlier_ratio_occ': outlier_ratio_occ,
                    'val_outlier_ratio_noc': outlier_ratio_noc
                }

            save_path = os.path.join(self.training_checkpoints_save_dir,
                                     self.experiment.ckpt_filename + '_' + str(self.global_step) + '.pytorch')

            torch.save(save_state_dict, save_path)

    def run_supervised(self):
        """ Executes the supervised training loop"""

        dataiter = iter(self.train_dataLoader)

        print(len(self.optimizer.param_groups[0]['params']))

        for end_step, lr in zip(self.step_list, self.lr_list):

            for param_group in self.optimizer.param_groups:
                param_group['lr'] = lr

            print('================================Learning Rate : {} =====================================================0'.
                  format(lr))

            while self.global_step <= end_step:
                try:
                    batch_input = next(dataiter)
                except StopIteration:
                    self.train_dataLoader = DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True,
                                                       drop_last=True, num_workers=self.num_workers, pin_memory=True)
                    dataiter = iter(self.train_dataLoader)
                    batch_input = next(dataiter)

                losses = supervised_loss(batch_input, module=self.moduleFlownets,
                                         params=self.params, return_flows=False,
                                         device_ids=self.device_ids, normalization=self.normalization)

                self.optimizer.zero_grad()
                losses.backward()
                self.optimizer.step()

                if self.global_step % self.logs_interval == 0:
                    self.train_SummaryWriter.add_scalar('Train_Loss', losses, self.global_step)
                    print('step {}/{} ========================================= loss: {}'.
                          format(self.global_step, self.params.get('num_iters'), losses))

                self.global_step += 1

            if self.network == 'flownet':
                self.moduleFlownets.eval()
            else:
                self.pwcnet.eval()

            flow_error_noc, flow_error_occ, outlier_ratio_noc, outlier_ratio_occ = self.eval()

            print('step: {}, ========================================= val_flow_error_noc: {}, '
                  'val_flow_error_occ: {}, outlier_ratio_noc: {}, outlier_ratio_occ: {}'.
                  format(self.global_step, flow_error_noc, flow_error_occ, outlier_ratio_noc,
                         outlier_ratio_occ))

            if self.network == 'flownet':
                if self.params.get('train_all'):
                    self.moduleFlownets.requires_grad_(requires_grad=True).train()
                else:
                    self.moduleFlownets[-1].requires_grad_(requires_grad=True).train()

                save_state_dict = {
                    'model_state_dict': self.moduleFlownets.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'global_step': self.global_step,
                    'val_flow_error_occ': flow_error_occ,
                    'val_flow_error_noc': flow_error_noc,
                    'val_outlier_ratio_occ': outlier_ratio_occ,
                    'val_outlier_ratio_noc': outlier_ratio_noc
                }

            else:
                self.pwcnet.requires_grad_(requires_grad=True).train()

                save_state_dict = {
                    'model_state_dict': self.pwcnet.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'global_step': self.global_step,
                    'val_flow_error_occ': flow_error_occ,
                    'val_flow_error_noc': flow_error_noc,
                    'val_outlier_ratio_occ': outlier_ratio_occ,
                    'val_outlier_ratio_noc': outlier_ratio_noc
                }

            save_path = os.path.join(self.training_checkpoints_save_dir,
                                     self.experiment.ckpt_filename + '_' + str(self.global_step) + '.pytorch')

            torch.save(save_state_dict, save_path)


    def eval(self):
        """ Performs the validation at the current state of the model"""

        dataiter = iter(self.eval_dataLoader)

        flow_error_noc = 0.0
        flow_error_occ = 0.0

        outlier_ratio_noc = 0.0
        outlier_ratio_occ = 0.0

        with torch.no_grad():
            for i in range(len(self.eval_dataLoader)):
                batch_input = next(dataiter)

                Image1 = batch_input['image1'].to(device=self.device_ids)
                Image2 = batch_input['image2'].to(device=self.device_ids)

                _, _, H, W = Image1.size()

                if self.params.get('val_height'):
                    eval_h = self.params.get('val_height')
                else:
                    eval_h = int((H // 64 + 1) * 64)

                if self.params.get('val_width'):
                    eval_w = self.params.get('val_width')
                else:
                    eval_w = int((W // 64 + 1) * 64)

                Image1 = (Image1 - self.normalization) / 255.0
                Image2 = (Image2 - self.normalization) / 255.0

                Image1 = torch.nn.functional.interpolate(Image1, size=(eval_h, eval_w), mode='bilinear')
                Image2 = torch.nn.functional.interpolate(Image2, size=(eval_h, eval_w), mode='bilinear')

                flow_noc = batch_input['flow_noc'].to(device=self.device_ids)
                flow_occ = batch_input['flow_occ'].to(device=self.device_ids)

                mask_noc = batch_input['mask_noc'].to(device=self.device_ids)
                mask_occ = batch_input['mask_occ'].to(device=self.device_ids)

                if self.network == 'flownet':
                    fw_flow = [self.moduleFlownets[0](Image1, Image2, None)]
                    for j in range(1, len(self.moduleFlownets)):
                        fw_flow.append(self.moduleFlownets[j](Image1, Image2, fw_flow[-1][0] * FLOW_SCALE * 4))
                    final_flow_fw = fw_flow[-1][0] * FLOW_SCALE * 4
                else:
                    fw_flow = self.pwcnet(Image1, Image2)
                    final_flow_fw = fw_flow[0] * FLOW_SCALE * 4

                flow_u, flow_v = torch.split(final_flow_fw, 1, 1)

                flow_u = torch.nn.functional.interpolate(flow_u, size=(H, W), mode='bilinear')
                flow_v = torch.nn.functional.interpolate(flow_v, size=(H, W), mode='bilinear')

                flow_u = flow_u * (W / eval_w)
                flow_v = flow_v * (H / eval_h)

                final_flow_fw = torch.cat((flow_u, flow_v), dim=1)

                flow_error_occ += flow_error_avg(final_flow_fw, flow_occ, mask_occ)
                flow_error_noc += flow_error_avg(final_flow_fw, flow_noc, mask_noc)

                outlier_ratio_occ += outlier_pct(flow_occ, final_flow_fw, mask_occ)
                outlier_ratio_noc += outlier_pct(flow_noc, final_flow_fw, mask_noc)

            flow_error_noc /= len(self.eval_dataLoader)
            flow_error_occ /= len(self.eval_dataLoader)
            outlier_ratio_occ /= len(self.eval_dataLoader)
            outlier_ratio_noc /= len(self.eval_dataLoader)

            self.eval_SummaryWriter.add_scalar('Flow_Error_NOC', flow_error_noc, self.global_step)
            self.eval_SummaryWriter.add_scalar('Flow_Error_ALL', flow_error_occ, self.global_step)
            self.eval_SummaryWriter.add_scalar('Outlier_Ratio_NOC', outlier_ratio_noc, self.global_step)
            self.eval_SummaryWriter.add_scalar('Outlier_Ratio_ALL', outlier_ratio_occ, self.global_step)

        return flow_error_noc, flow_error_occ, outlier_ratio_noc, outlier_ratio_occ




























