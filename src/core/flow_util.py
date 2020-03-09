import torch
import numpy as np
from skimage.color import hsv2rgb
import torch.nn.functional as F
import copy
import png

def flow_to_color(flow, mask=None, max_flow=None):
    """Converts flow to 3-channel color image.
    Args:
        flow: tensor of shape [num_batch, 2, height, width].
        mask: flow validity mask of shape [num_batch, 1, height, width].
    """

    n = 8
    B, _, H, W = flow.size()
    mask = torch.ones(B, 1, H, W, dtype=flow.dtype, device=flow.device) \
        if mask is None else mask

    flow_u, flow_v = torch.split(flow, 1, dim=1)

    if max_flow is not None:
        max_flow = torch.max(torch.tensor(max_flow), torch.tensor(1.0))
    else:
        max_flow = torch.max(torch.abs(flow * mask))

    mag = torch.pow(torch.sum(torch.pow(flow, 2), dim=1, keepdim=True), 0.5)
    angle = torch.atan2(flow_v, flow_u)

    im_h = torch.fmod(angle / (2 * np.pi) + 1.0, 1.0)
    im_s = torch.clamp(mag * n / max_flow, 0., 1.)
    im_v = torch.clamp(n - im_s, 0., 1.)

    im_hsv = torch.cat((im_h, im_s, im_v), dim=1)

    im_hsv = im_hsv.permute(0, 2, 3, 1)

    im_rgb = np.empty((B, H, W, 3))

    for i in range(B):
        im_rgb[i, :, :, :] = hsv2rgb(im_hsv[i, :, :, :].cpu().numpy())

    return torch.tensor(im_rgb, dtype=im_hsv.dtype).permute(0, 3, 1, 2)


def read_flow_png(filename):
    flow_object = png.Reader(filename=filename)
    flow_direct = flow_object.asDirect()
    flow_data = list(flow_direct[2])
    (w, h) = flow_direct[3]['size']
    flow = np.zeros((h, w, 3), dtype=np.float32)

    for i in range(len(flow_data)):
        flow[i, :, 0] = flow_data[i][0::3]
        flow[i, :, 1] = flow_data[i][1::3]
        flow[i, :, 2] = flow_data[i][2::3]

    return flow

def flow_error_image(flow_1, flow_2, mask_occ, mask_noc=None, log_colors=True):

    """Visualize the error between two flows as 3-channel color image. """

    B ,C, H, W = flow_1.size()
    diff_sq = torch.pow(flow_1 - flow_2, 2)
    diff = torch.pow(torch.sum(diff_sq, dim=1, keepdim=True), 0.5)

    mask_noc = torch.ones(B, C, H, W, dtype=flow_1.dtype, device=flow_1.device) \
         if mask_noc is None else mask_noc

    error = (torch.clamp(diff, 0.0, 5.0) / 5.0) * mask_occ
    img_g = error * mask_noc
    img_b = error * mask_noc

    error_img = torch.cat((error, img_g, img_b), dim=1)

    return error_img

def flow_error_avg(flow_1, flow_2, mask):
    """Evaluates the average endpoint error between flow batches."""

    diff = euclidean(flow_1 - flow_2) * mask
    return torch.sum(diff) / torch.sum(mask)


def outlier_ratio(gt_flow, flow, mask, threshold=3.0, relative=0.05):
    """ Evcaluate the F1 metric between estimated and ground truth flow"""

    diff = euclidean(gt_flow - flow) * mask

    if relative is not None:
        threshold = torch.max(torch.tensor(threshold, device=flow.device), euclidean(gt_flow) * relative)
        outliers = (diff >= threshold)
    else:
        outliers = (diff >= threshold)

    outlier_ratio = torch.sum(outliers) / torch.sum(mask)

    return outlier_ratio


def outlier_pct(gt_flow, flow, mask, threshold=3.0, relative=0.05):
    return outlier_ratio(gt_flow, flow, mask, threshold=3.0, relative=0.05) * 100


def euclidean(x):
    return torch.pow(torch.sum(torch.pow(x, 2), dim=1, keepdim=True), 0.5)

def resize_output_flow(flow, scale_factor=None, height=None, width=None):

    B, C, H, W = flow.size()

    if scale_factor is not None:
        h = int(H * scale_factor)
        w = int(W * scale_factor)
    else:
        h = copy.deepcopy(height)
        w = copy.deepcopy(width)

    resized_flow = F.interpolate(flow, size=(h, w), mode='bilinear')
    u, v = torch.split(resized_flow, 1, 1)
    u = u * w / W
    v = v * h / H

    return torch.cat((u, v), dim=1)




