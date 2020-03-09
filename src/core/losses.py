import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from src.core.image_warp import image_warp


def compute_losses(im1, im2, flow_fw, flow_bw,
                   border_mask=None,
                   mask_occlusion='',
                   data_max_distance=1):
    """ Returns the bidirectional loss as a dictionary with leys referring to specific loss types"""

    losses = {}

    im2_warped = image_warp(im2, flow_fw)
    im1_warped = image_warp(im1, flow_bw)

    flow_bw_warped = image_warp(flow_bw, flow_fw)
    flow_fw_warped = image_warp(flow_fw, flow_bw)

    flow_diff_fw = flow_fw + flow_bw_warped
    flow_diff_bw = flow_bw + flow_fw_warped

    im_diff_fw = im1 - im2_warped
    im_diff_bw = im2 - im1_warped

    if border_mask is None:
        mask_fw = create_outgoing_mask(flow_fw)
        mask_bw = create_outgoing_mask(flow_bw)
    else:
        mask_fw = border_mask.clone().detach()
        mask_bw = border_mask.clone().detach()

    fb_occ_fw = occlusion(flow_fw, flow_bw_warped).requires_grad_(True)
    fb_occ_bw = occlusion(flow_bw, flow_fw_warped).requires_grad_(True)

    if mask_occlusion:
        mask_fw *= (1.0 - fb_occ_fw)
        mask_bw *= (1.0 - fb_occ_bw)

    occ_fw = 1.0 - mask_fw
    occ_bw = 1.0 - mask_bw

    losses['photo'] = (photometric_loss(im_diff_fw, mask_fw) +
                        photometric_loss(im_diff_bw, mask_bw))

    losses['occ'] = (charbonnier_loss(occ_fw) +
                     charbonnier_loss(occ_bw))

    grad_loss_fw = gradient_loss(im1, im2_warped, mask_fw)
    grad_loss_bw = gradient_loss(im2, im1_warped, mask_bw)
    losses['grad'] = grad_loss_fw + grad_loss_bw

    smoothness_loss_fw = smoothness_loss(flow_fw)
    smoothness_loss_bw = smoothness_loss(flow_bw)
    losses['smooth_1st'] = smoothness_loss_fw + smoothness_loss_bw

    secomd_order_loss_fw = second_order_loss(flow_fw)
    secomd_order_loss_bw = second_order_loss(flow_bw)
    losses['smooth_2nd'] = secomd_order_loss_fw + secomd_order_loss_bw

    second_order_edge_aware_loss_fw = second_order_edge_aware_loss(flow_fw, im1)
    second_order_edge_aware_loss_bw = second_order_edge_aware_loss(flow_bw, im2)
    losses['smooth_2nd_edge'] = second_order_edge_aware_loss_fw + second_order_edge_aware_loss_bw

    losses['fb'] = (charbonnier_loss(flow_diff_fw, mask_fw) +
                    charbonnier_loss(flow_diff_bw, mask_bw))

    losses['ternary'] = ternary_loss(im1, im2_warped, mask_fw, max_distance=data_max_distance) \
                        + ternary_loss(im2, im1_warped, mask_bw, max_distance=data_max_distance)

    return losses


def occlusion(flow_fw, flow_bw_warped):
    """ Returns the occlusion based on forward flow and warped backward flow """

    flow_diff_fw = flow_fw + flow_bw_warped

    mag_sq_fw = torch.sum(torch.pow(flow_fw, 2), 1, keepdim=True) + torch.sum(torch.pow(flow_bw_warped, 2), 1, keepdim=True)
    occ_thresh_fw = 0.01 * mag_sq_fw + 0.5

    fb_occ_fw = (torch.sum(torch.pow(flow_diff_fw, 2), 1, keepdim=True) > occ_thresh_fw).to(dtype=flow_fw.dtype)

    return fb_occ_fw.to(dtype=flow_fw.dtype)


def create_outgoing_mask(flow):
    """ Creates mask for pixels whose flow exceeds the image boundaries """

    B, C, H, W = flow.size()

    grid_x = torch.arange(0, W, dtype=flow.dtype, device=flow.device).view(1, -1).repeat(B, H, 1)
    grid_y = torch.arange(0, H, dtype=flow.dtype, device=flow.device).view(-1, 1).repeat(B, 1, W)

    flow_u, flow_v = torch.split(flow, 1, dim=1)

    pos_x = flow_u + grid_x
    pos_y = flow_v + grid_y

    inside_x = (pos_x <= (W - 1)) & (pos_x >= 0)
    inside_y = (pos_y <= (H - 1)) & (pos_y >= 0)

    inside = inside_x & inside_y

    outgoing_mask = torch.unsqueeze(inside, dim=1)

    return outgoing_mask


def photometric_loss(im_diff, mask):
    return charbonnier_loss(im_diff, mask, beta=255)


def charbonnier_loss(x, mask=None, truncate=None, alpha=0.45, beta=1.0, epsilon=0.001):

    B, C, H, W = x.size()

    normalization = B * H * W * C

    error = torch.pow(torch.pow(x * beta, 2) + epsilon**2, alpha)

    if mask is not None:
        error = error * mask

    loss = torch.sum(error) / normalization

    return loss

def gradient_loss(im1, im2_warped, mask):

    mask_x = create_mask(im1, [[0, 0], [1, 1]])
    mask_y = create_mask(im1, [[1, 1], [0, 0]])

    grad_mask = torch.cat((mask_x, mask_y), dim=1).repeat(1, 3, 1, 1)

    diff = _gradient_delta(im1, im2_warped)
    mask_broadcast = mask.repeat(1, 6, 1, 1)

    grad_loss = charbonnier_loss(diff, grad_mask * mask_broadcast)

    return grad_loss

def _gradient_delta(im1, im2_warped):

    filter_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    filter_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])

    weight_array = np.zeros((6, 3, 3, 3))

    for c in range(3):
        weight_array[2*c, c, :, :] = filter_x
        weight_array[2*c + 1, c, :, :] = filter_y

    weights = torch.tensor(weight_array, dtype=im1.dtype, device=im1.device)

    im1_grad = F.conv2d(im1, weights, padding=1)
    im2_warped_grad = F.conv2d(im2_warped, weights, padding=1)

    diff = im1_grad - im2_warped_grad

    return diff


def _smoothness_delta(flow):

    mask_x = create_mask(flow, [[0, 0], [0, 1]])
    mask_y = create_mask(flow, [[0, 1], [0, 0]])

    mask = torch.cat((mask_x, mask_y), dim=1)

    filter_x = np.array([[0, 0, 0], [0, 1, -1], [0, 0, 0]])
    filter_y = np.array([[0, 0, 0], [0, 1, 0], [0, -1, 0]])

    weights_array = np.zeros((2, 1, 3, 3))

    weights_array[0, 0, :, :] = filter_x
    weights_array[1, 0, :, :] = filter_y

    weights = torch.tensor(weights_array, dtype=flow.dtype, device=flow.device)

    flow_u, flow_v = torch.split(flow, 1, dim=1)

    delta_u = F.conv2d(flow_u, weights, padding=1)
    delta_v = F.conv2d(flow_v, weights, padding=1)

    return delta_u, delta_v, mask


def _second_order_deltas(flow):

    mask_x = create_mask(flow, [[0, 0], [1, 1]])
    mask_y = create_mask(flow, [[1, 1], [0, 0]])
    mask_diag = create_mask(flow, [[1, 1], [1, 1]])

    mask = torch.cat((mask_x, mask_y, mask_diag, mask_diag), dim=1)

    filter_x = np.array([[0, 0, 0], [1, -2, 1], [0, 0, 0]])
    filter_y = np.array([[0, 1, 0], [0, -2, 0], [0, 1, 0]])
    filter_diag1 = np.array([[1, 0, 0], [0, -2, 0], [0, 0, 1]])
    filter_diag2 = np.array([[0, 0, 1], [0, -2, 0], [1, 0, 0]])

    weights_array = np.zeros((4, 1, 3, 3))

    weights_array[0, 0, :, :] = filter_x
    weights_array[1, 0, :, :] = filter_y
    weights_array[2, 0, :, :] = filter_diag1
    weights_array[3, 0, :, :] = filter_diag2

    weights = torch.tensor(weights_array, dtype=flow.dtype, device=flow.device)

    flow_u, flow_v = torch.split(flow, 1, dim=1)

    delta_u = F.conv2d(flow_u, weights, padding=1)
    delta_v = F.conv2d(flow_v, weights, padding=1)

    return delta_u, delta_v, mask


def second_order_loss(flow):

    delta_u, delta_v, mask = _second_order_deltas(flow)

    loss_u = charbonnier_loss(delta_u, mask)
    loss_v = charbonnier_loss(delta_v, mask)

    loss = loss_u + loss_v

    return loss

def second_order_edge_aware_loss(flow, img):

    r, g, b = torch.split(img, 1, 1)
    image = (0.2989 * r + 0.5870 * g + 0.1140 * b)

    delta_u, delta_v, mask = _second_order_deltas(flow)

    filter_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    filter_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])

    weights_array = np.zeros((2, 1, 3, 3))

    weights_array[0, 0, :, :] = filter_x
    weights_array[1, 0, :, :] = filter_y

    weights = torch.tensor(weights_array, dtype=flow.dtype, device=flow.device)

    mask_grad = create_mask(flow, [[1, 1], [1, 1]])
    image_grad = torch.sqrt(torch.sum(torch.pow(F.conv2d(image, weights, padding=1), 2), dim=1, keepdim=True))

    delta_u_with_grad = delta_u * torch.exp(-3.0 * image_grad)
    delta_v_with_grad = delta_v * torch.exp(-3.0 * image_grad)

    loss_u = charbonnier_loss(delta_u_with_grad, mask * mask_grad)
    loss_v = charbonnier_loss(delta_v_with_grad, mask * mask_grad)

    loss = loss_u + loss_v

    return loss


def smoothness_loss(flow):

    delta_u, delta_v, mask = _smoothness_delta(flow)

    loss_u = charbonnier_loss(delta_u, mask)
    loss_v = charbonnier_loss(delta_v, mask)

    loss = loss_u + loss_v

    return loss


def create_mask(x, paddings):

    B, C, H, W = x.size()

    inner_height = H - (paddings[0][0] + paddings[0][1])
    inner_width = W - (paddings[1][0] + paddings[1][1])

    inner = torch.ones(inner_height, inner_width, dtype=x.dtype, device=x.device)

    mask2d = F.pad(inner, [paddings[1][0], paddings[1][1], paddings[0][0], paddings[0][1]])
    mask3d = torch.unsqueeze(mask2d, dim=0).repeat(B, 1, 1)
    mask4d = torch.unsqueeze(mask3d, dim=1)
    mask4d.requires_grad = False

    return mask4d


def create_border_mask(x, border_ratio=0.1):

    B, C, H, W = x.size()

    min_dim = H if H <= W else W
    sz = np.ceil(min_dim * border_ratio)
    border_mask = create_mask(x, [[sz, sz], [sz, sz]])

    return border_mask


def ternary_loss(im1, im2_warped, mask, max_distance=1):

    patch_size = 2 * max_distance + 1

    def _ternary_transform(image):

        # intensities = (torch.sum(image, dim=1, keepdim=True) / 3) * 255
        r, g, b = torch.split(image, 1, 1)
        intensities = (0.2989 * r + 0.5870 * g + 0.1140 * b) * 255
        out_channels = patch_size * patch_size
        w = np.eye(out_channels).reshape((out_channels, 1, patch_size, patch_size))

        weights = torch.tensor(w, dtype=torch.float32, device=image.device)
        patches = F.conv2d(intensities, weights, stride=1, padding=max_distance)

        # import pdb; pdb.set_trace()
        transf = patches - intensities
        transf_norm = transf / torch.pow(torch.pow(transf, 2) + 0.81, 0.5)

        return transf_norm

    def _hamming_distance(t1, t2):

        dist = torch.pow(t1 - t2, 2)
        dist_norm = dist / (0.1 + dist)
        dist_sum = torch.sum(dist_norm, dim=1, keepdim=True)
        return dist_sum

    t1 = _ternary_transform(im1)
    t2 = _ternary_transform(im2_warped)

    dist = _hamming_distance(t1, t2)

    transform_mask = create_mask(mask, [[max_distance, max_distance],
                                        [max_distance, max_distance]])
    return charbonnier_loss(dist, mask * transform_mask)