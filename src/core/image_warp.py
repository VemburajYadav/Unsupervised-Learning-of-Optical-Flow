import torch
import torch.nn as nn
import torch.nn.functional as F


def image_warp(input, flow):
    """ Warps the input tensor using the flow tensor by bilinear interpolation """

    B, C, H, W = input.size()

    xx = torch.arange(0, W, dtype=flow.dtype, device=flow.device).view(1, -1).repeat(H, 1)
    yy = torch.arange(0, H, dtype=flow.dtype, device=flow.device).view(-1, 1).repeat(1, W)

    xx = xx.view(1, 1, H, W).repeat(B, 1, 1, 1)
    yy = yy.view(1, 1, H, W).repeat(B, 1, 1, 1)

    flow_u, flow_v = torch.split(flow, 1, 1)

    xx_new = xx + flow_u
    yy_new = yy + flow_v

    flow_u_norm = 2.0 * xx_new / (W - 1) - 1.0
    flow_v_norm = 2.0 * yy_new / (H - 1) - 1.0

    vgrid = torch.cat((flow_u_norm, flow_v_norm), dim=1)

    vgrid = vgrid.permute(0, 2, 3, 1).contiguous()

    input_warp = F.grid_sample(input, vgrid, padding_mode='border')

    return input_warp