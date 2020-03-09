import torch
from src.core.losses import compute_losses
from src.core.flownet import FLOW_SCALE

def unsupervised_loss_flownet(batch_input, module=None, loss_weights_dict=None,
                      params=None, return_flow=False, normalization=None, device_ids=0):
    """ Computes the weighted sum of different losses at different pyramid lvevls for flownet """

    full_res = params.get('full_res')

    normalization = normalization / 255
    pyramid_loss = params.get('pyramid_loss')

    Image1 = batch_input['image1'].to(device=torch.device('cuda:0'))
    Image2 = batch_input['image2'].to(device=torch.device('cuda:0'))

    Image1 = Image1 / 255
    Image2 = Image2 / 255

    if params.get('border_mask'):
        border_mask = batch_input['border_mask'].to(device=torch.device('cuda:0'))
    else:
        border_mask = None

    Image1_norm = Image1 - normalization
    Image2_norm = Image2 - normalization

    fw_flow = [module[0](Image1_norm, Image2_norm, None)]
    bw_flow = [module[0](Image2_norm, Image1_norm, None)]

    for i in range(1, len(module)):
        fw_flow.append(module[i](Image1_norm, Image2_norm, fw_flow[-1][0] * FLOW_SCALE * 4))
        bw_flow.append(module[i](Image2_norm, Image1_norm, bw_flow[-1][0] * FLOW_SCALE * 4))

    flow_fw = fw_flow[-1]
    flow_bw = bw_flow[-1]

    layer_weights = [12.7, 4.35, 3.9, 3.4, 1.1]
    layer_patch_distances = [3, 2, 2, 1, 1]

    im1_s = Image1.clone().detach()
    im2_s = Image2.clone().detach()
    mask_s = border_mask.clone().detach()

    if full_res:
        layer_weights = [12.7, 5.5, 5.0, 4.35, 3.9, 3.4, 1.1]
        layer_patch_distances = [3, 3] + layer_patch_distances
        final_flow_scale = FLOW_SCALE * 4
        scale_factor = 1.0
        final_flow_fw = flow_fw[0] * final_flow_scale
        final_flow_bw = flow_bw[0] * final_flow_scale
        flow_pair = zip(flow_fw, flow_bw)
    else:
        final_flow_scale = FLOW_SCALE
        scale_factor = 0.25
        final_flow_fw = flow_fw[0] * final_flow_scale * 4
        final_flow_bw = flow_bw[0] * final_flow_scale * 4
        flow_pair = zip(flow_fw[2:], flow_bw[2:])

    if pyramid_loss:
        flow_enum = enumerate(flow_pair)
    else:
        flow_enum = [(0, (flow_fw[0], flow_bw[0]))]

    mask_occlusion = params.get('mask_occlusion', '')

    combined_loss = 0.0

    for i, flow in flow_enum:
        flow_scale = final_flow_scale / (2 ** i)
        flow_fw_s, flow_bw_s = flow

        im1_s = torch.nn.functional.interpolate(im1_s, scale_factor=scale_factor, mode='bilinear')
        im2_s = torch.nn.functional.interpolate(im2_s, scale_factor=scale_factor, mode='bilinear')
        mask_s = torch.nn.functional.interpolate(mask_s, scale_factor=scale_factor, mode='bilinear')

        losses = compute_losses(im1_s, im2_s,
                                flow_fw_s * flow_scale, flow_bw_s * flow_scale,
                                border_mask=mask_s if params.get('border_mask') else None,
                                mask_occlusion=mask_occlusion,
                                data_max_distance=layer_patch_distances[i])

        layer_loss = 0.0

        for loss_type, loss_weight in loss_weights_dict.items():
            name = loss_type.rstrip('_weight')
            layer_loss += loss_weight * losses[name]

        combined_loss += layer_weights[i] * layer_loss

        scale_factor = 0.5

    if not return_flow:
        return combined_loss

    return combined_loss, final_flow_fw, final_flow_bw


def unsupervised_loss_pwcnet(batch_input, module=None, loss_weights_dict=None,
                      params=None, return_flow=False, normalization=None, device_ids=0):
    """ Computes the weighted sum of different losses at different pyramid lvevls for pwcnet """

    full_res = params.get('full_res')

    normalization = normalization / 255
    pyramid_loss = params.get('pyramid_loss')

    Image1 = batch_input['image1'].to(device=torch.device('cuda:0'))
    Image2 = batch_input['image2'].to(device=torch.device('cuda:0'))

    Image1 = Image1 / 255
    Image2 = Image2 / 255

    if params.get('border_mask'):
        border_mask = batch_input['border_mask'].to(device=torch.device('cuda:0'))
    else:
        border_mask = None

    Image1_norm = Image1 - normalization
    Image2_norm = Image2 - normalization

    flow_fw = module(Image1_norm, Image2_norm)
    flow_bw = module(Image2_norm, Image1_norm)

    layer_weights = [12.7, 4.35, 3.9, 3.4, 1.1]
    layer_patch_distances = [3, 2, 2, 1, 1]

    im1_s = Image1.clone().detach()
    im2_s = Image2.clone().detach()
    mask_s = border_mask.clone().detach()

    if full_res:
        layer_weights = [12.7, 5.5, 5.0, 4.35, 3.9, 3.4, 1.1]
        layer_patch_distances = [3, 3] + layer_patch_distances
        final_flow_scale = FLOW_SCALE * 4
        scale_factor = 1.0
        final_flow_fw = flow_fw[0] * final_flow_scale
        final_flow_bw = flow_bw[0] * final_flow_scale
        flow_pair = zip(flow_fw, flow_bw)
    else:
        final_flow_scale = FLOW_SCALE
        scale_factor = 0.25
        final_flow_fw = flow_fw[0] * final_flow_scale * 4
        final_flow_bw = flow_bw[0] * final_flow_scale * 4
        flow_pair = zip(flow_fw[2:], flow_bw[2:])

    if pyramid_loss:
        flow_enum = enumerate(flow_pair)
    else:
        flow_enum = [(0, (flow_fw[0], flow_bw[0]))]

    mask_occlusion = params.get('mask_occlusion', '')

    combined_loss = 0.0

    for i, flow in flow_enum:
        flow_scale = final_flow_scale / (2 ** i)
        flow_fw_s, flow_bw_s = flow

        im1_s = torch.nn.functional.interpolate(im1_s, scale_factor=scale_factor, mode='bilinear')
        im2_s = torch.nn.functional.interpolate(im2_s, scale_factor=scale_factor, mode='bilinear')
        mask_s = torch.nn.functional.interpolate(mask_s, scale_factor=scale_factor, mode='bilinear')

        losses = compute_losses(im1_s, im2_s,
                                flow_fw_s * flow_scale, flow_bw_s * flow_scale,
                                border_mask=mask_s if params.get('border_mask') else None,
                                mask_occlusion=mask_occlusion,
                                data_max_distance=layer_patch_distances[i])

        layer_loss = 0.0

        for loss_type, loss_weight in loss_weights_dict.items():
            name = loss_type.rstrip('_weight')
            layer_loss += loss_weight * losses[name]

        combined_loss += layer_weights[i] * layer_loss

        scale_factor = 0.5

    if not return_flow:
        return combined_loss

    return combined_loss, final_flow_fw, final_flow_bw















