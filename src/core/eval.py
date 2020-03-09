import os
from PIL import Image
import numpy as np
import torch
import glob
from src.core.flownet import Simple, Complex
from src.core.flow_util import read_flow_png, resize_output_flow, flow_to_color, flow_error_avg, outlier_pct, flow_error_image
from src.core.losses import occlusion
from src.core.image_warp import image_warp
from skimage import io
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--logdir', type=str, default='../../log/C_Gradient_KITTI/Final_CKPTs', help='Path to the checkpoint directory')
parser.add_argument('--ckpt_file', type=str, default='ckpt_402001.pytorch', help='Checkpoint file')
parser.add_argument('--arch', type=str, default='C', help="architecture: 'C' or 'CS' ")
parser.add_argument('--img_1_dir', type=str, default='../../../KITTI_Optical_flow/data_scene_flow/training/image_2',
                    help='Directory for images as first frame')
parser.add_argument('--img_2_dir', type=str, default='../../../KITTI_Optical_flow/data_scene_flow/training/image_3',
                    help='Directory for images as second frames')
parser.add_argument('--flow_occ_dir', type=str, default='../../../KITTI_Optical_flow/data_scene_flow/training/flow_occ',
                    help='Directory with ground truth flow for all pixels')
parser.add_argument('--flow_noc_dir', type=str, default='../../../KITTI_Optical_flow/data_scene_flow/training/flow_noc',
                    help='Directory with ground truth flow for only non occluded pixels')
parser.add_argument('--save_dir', type=str, default='../../../Eval_Outputs/C_Grad_KITTI',
                    help='Path of the directory to save the predicted flows, occlusion maps and flow error maps')


opt = parser.parse_args()

logdir = opt.logdir
ckpt_file = os.path.join(logdir, opt.ckpt_file)
arch = opt.arch
img_1_dir = opt.img_1_dir
img_2_dir = opt.img_2_dir
flow_occ_dir = opt.flow_occ_dir
flow_noc_dir = opt.flow_noc_dir
save_dir = opt.save_dir

channel_mean = np.array([104.920005,110.1753,114.785955], dtype=np.float32)

flow_dir = os.path.join(save_dir, 'flow')
flow_gt_dir = os.path.join(save_dir, 'flow_gt')
occ_map_dir = os.path.join(save_dir, 'occ')
occ_map_gt_dir = os.path.join(save_dir, 'occ_gt')
flow_error_dir = os.path.join(save_dir, 'flow_error')

if not os.path.isdir(flow_dir):
    os.makedirs(flow_dir)
if not os.path.isdir(flow_gt_dir):
    os.makedirs(flow_gt_dir)
if not os.path.isdir(occ_map_dir):
    os.makedirs(occ_map_dir)
if not os.path.isdir(occ_map_gt_dir):
    os.makedirs(occ_map_gt_dir)
if not os.path.isdir(flow_error_dir):
    os.makedirs(flow_error_dir)

moduleList = []

for module in arch:
    if module == 'C':
        moduleList.append(Complex())
    elif module == 'S':
        moduleList.append(Simple())

moduleFlownets = torch.nn.ModuleList(moduleList)
state_dict = torch.load(ckpt_file)['model_state_dict']
moduleFlownets.load_state_dict(state_dict)

moduleFlownets = moduleFlownets.to(device=0)

img_1_fnames = sorted(glob.glob(os.path.join(img_1_dir, '*_10.png')))
img_2_fnames = sorted(glob.glob(os.path.join(img_1_dir, '*_11.png')))

N = len(img_1_fnames)

flow_error_noc_sum = 0.0
flow_error_occ_sum = 0.0
flow_outlier_noc_sum = 0.0
flow_outlier_occ_sum = 0.0

for i in range(N):
    img_1_fname = img_1_fnames[i].split('/')[-1]
    img_2_fname = img_2_fnames[i].split('/')[-1]

    img1_path = os.path.join(img_1_dir, img_1_fname)
    img2_path = os.path.join(img_1_dir, img_2_fname)
    flow_occ_path = os.path.join(flow_occ_dir, img_1_fname)
    flow_noc_path = os.path.join(flow_noc_dir, img_1_fname)

    img1 = np.array(Image.open(img1_path), dtype=np.float32)
    img2 = np.array(Image.open(img2_path), dtype=np.float32)

    if img1.ndim == 2:
        img1 = np.tile(np.reshape(img1, (img1.shape[0], img1.shape[1], 1)), (1, 1, 3))
        img2 = np.tile(np.reshape(img2, (img2.shape[0], img2.shape[1], 1)), (1, 1, 3))

    img1 = (img1 - channel_mean) / 255.0
    img2 = (img2 - channel_mean) / 255.0

    flow_noc = read_flow_png(flow_noc_path)
    flow_occ = read_flow_png(flow_occ_path)

    mask_noc = np.expand_dims(flow_noc[:, :, 2], 2)
    mask_occ = np.expand_dims(flow_occ[:, :, 2], 2)

    flow_noc = (flow_noc[:, :, 0:2] - 2**15) / 64.0
    flow_occ = (flow_occ[:, :, 0:2] - 2**15) / 64.0

    img1 = torch.FloatTensor(np.expand_dims(img1, 0)).permute(0,3,1,2).contiguous().to(device=0)
    img2 = torch.FloatTensor(np.expand_dims(img2, 0)).permute(0,3,1,2).contiguous().to(device=0)
    flow_noc = torch.FloatTensor(np.expand_dims(flow_noc, 0)).permute(0,3,1,2).contiguous().to(device=0)
    flow_occ = torch.FloatTensor(np.expand_dims(flow_occ, 0)).permute(0,3,1,2).contiguous().to(device=0)
    mask_noc = torch.FloatTensor(np.expand_dims(mask_noc, 0)).permute(0,3,1,2).contiguous().to(device=0)
    mask_occ = torch.FloatTensor(np.expand_dims(mask_occ, 0)).permute(0,3,1,2).contiguous().to(device=0)

    B, _, H, W = img1.size()

    eval_h = int((H // 64 + 1) * 64)
    eval_w = int((W // 64 + 1) * 64)

    img1 = torch.nn.functional.interpolate(img1, size=(eval_h, eval_w), mode='bilinear')
    img2 = torch.nn.functional.interpolate(img2, size=(eval_h, eval_w), mode='bilinear')

    with torch.no_grad():
        fw_flow = [moduleFlownets[0](img1, img2, None)]
        bw_flow = [moduleFlownets[0](img2, img1, None)]

        for j in range(1, len(moduleFlownets)):
            fw_flow.append(moduleFlownets[j](img1, img2, fw_flow[-1][0] * 5 * 4))
            bw_flow.append(moduleFlownets[j](img2, img1, bw_flow[-1][0] * 5 * 4))

    final_flow_fw = fw_flow[-1][0] * 5 * 4
    final_flow_bw = bw_flow[-1][0] * 5 * 4

    flow_u, flow_v = torch.split(final_flow_fw, 1, 1)
    flow_u_bw, flow_v_bw = torch.split(final_flow_bw, 1, 1)

    flow_u = torch.nn.functional.interpolate(flow_u, size=(H, W), mode='bilinear')
    flow_v = torch.nn.functional.interpolate(flow_v, size=(H, W), mode='bilinear')

    flow_u_bw = torch.nn.functional.interpolate(flow_u_bw, size=(H, W), mode='bilinear')
    flow_v_bw = torch.nn.functional.interpolate(flow_v_bw, size=(H, W), mode='bilinear')

    flow_u = flow_u * (W / eval_w)
    flow_v = flow_v * (H / eval_h)

    flow_u_bw = flow_u_bw * (W / eval_w)
    flow_v_bw = flow_v_bw * (H / eval_h)

    final_flow_fw = torch.cat((flow_u, flow_v), dim=1)
    final_flow_bw = torch.cat((flow_u_bw, flow_v_bw), dim=1)

    flow_error_occ = flow_error_avg(final_flow_fw, flow_occ, mask_occ)
    flow_error_noc = flow_error_avg(final_flow_fw, flow_noc, mask_noc)

    flow_error_noc_sum = flow_error_noc_sum + flow_error_noc
    flow_error_occ_sum = flow_error_occ_sum + flow_error_occ

    flow_outlier_occ = outlier_pct(flow_occ, final_flow_fw, mask_occ)
    flow_outlier_noc = outlier_pct(flow_noc, final_flow_fw, mask_noc)

    flow_outlier_noc_sum = flow_outlier_noc_sum + flow_outlier_noc
    flow_outlier_occ_sum = flow_outlier_occ_sum + flow_outlier_occ

    flow_bw_warped = image_warp(final_flow_bw, final_flow_fw)
    flow_fw_warped = image_warp(final_flow_fw, final_flow_bw)

    occ_mask_fw = occlusion(final_flow_fw, flow_bw_warped) * mask_occ
    occ_mask_bw = occlusion(final_flow_bw, flow_fw_warped)

    occ_mask_fw_gt = mask_occ - mask_noc

    flow_fw_color = flow_to_color(final_flow_fw)
    flow_fw_color_gt = flow_to_color(flow_occ, mask_occ)

    flow_fw_error_img = flow_error_image(flow_occ, final_flow_fw, mask_occ,
                                         mask_noc=mask_noc)

    flow_fw_color_np = (flow_fw_color.permute(0,2,3,1).cpu().numpy()[0, :, :, :] * 255).astype(np.uint8)
    flow_fw_color_gt_np = (flow_fw_color_gt.permute(0,2,3,1).cpu().numpy()[0, :, :, :] * 255).astype(np.uint8)

    flow_fw_error_img_np = (flow_fw_error_img.permute(0,2,3,1).cpu().numpy()[0, :, :, :] * 255).astype(np.uint8)

    occ_fw_np = (occ_mask_fw.permute(0,2,3,1).cpu().numpy()[0, :, :, :] * 255).astype(np.uint8)
    occ_bw_np = (occ_mask_bw.permute(0,2,3,1).cpu().numpy()[0, :, :, :] * 255).astype(np.uint8)
    occ_fw_gt_np = (occ_mask_fw_gt.permute(0,2,3,1).cpu().numpy()[0, :, :, :] * 255).astype(np.uint8)

    occ_fw_np = np.tile(occ_fw_np, (1, 1, 3))
    occ_bw_np = np.tile(occ_bw_np, (1, 1, 3))
    occ_fw_gt_np = np.tile(occ_fw_gt_np, (1, 1, 3))

    io.imsave(os.path.join(flow_dir, img_1_fname), flow_fw_color_np)
    io.imsave(os.path.join(flow_gt_dir, img_1_fname), flow_fw_color_gt_np)
    io.imsave(os.path.join(flow_error_dir, img_1_fname), flow_fw_error_img_np)
    io.imsave(os.path.join(occ_map_gt_dir, img_1_fname), occ_fw_gt_np)
    io.imsave(os.path.join(occ_map_dir, img_1_fname), occ_fw_np)

flow_error_noc_avg = flow_error_noc_sum / N
flow_error_occ_avg = flow_error_occ_sum / N
flow_outlier_noc_avg = flow_outlier_noc_sum / N
flow_outlier_occ_avg = flow_outlier_occ_sum / N

print('EPE OCC: {}'.format(flow_error_occ_avg))
print('EPE NOC: {}'.format(flow_error_noc_avg))
print('Fl OCC: {}'.format(flow_outlier_occ_avg))
print('Fl NOC: {}'.format(flow_outlier_noc_avg))




