import torch
from src.core.losses import charbonnier_loss
from src.core.flownet import FLOW_SCALE
from src.core.flow_util import resize_output_flow


def supervised_loss(batch_input, module=None,
                    params=None, return_flows=False, device_ids=0, normalization=None):

    full_res = params.get('full_res')

    normalization = normalization / 255

    Image1 = batch_input['image1'].to(device=device_ids)
    Image2 = batch_input['image2'].to(device=device_ids)

    flow_gt = batch_input['flow_occ'].to(device=device_ids)
    mask_gt = batch_input['mask_occ'].to(device=device_ids)

    Image1 = Image1 / 255
    Image2 = Image2 / 255

    Image1_norm = Image1 - normalization
    Image2_norm = Image2 - normalization

    fw_flow = [module[0](Image1_norm, Image2_norm, None)]

    for i in range(1, len(module)):
        fw_flow.append(module[i](Image1_norm, Image2_norm, fw_flow[-1][0] * FLOW_SCALE * 4))

    if params.get('full_res'):
        if params.get('train_all'):
            flow_fw = [x[0] for x in fw_flow]
        else:
            flow_fw = [fw_flow[-1][0]]
    else:
        if params.get('train_all'):
            flow_fw = [x[2] for x in fw_flow]
        else:
            flow_fw = [fw_flow[-1][2]]

        mask_gt = torch.nn.functional.interpolate(mask_gt, scale_factor=0.25, mode='bilinear')
        flow_gt = resize_output_flow(flow_gt, scale_factor=0.25)

    if params.get('border_mask'):
        border_mask = batch_input['border_mask'].to(device=device_ids)
        if not params.get('full_res'):
            border_mask = torch.nn.functional.interpolate(border_mask, scale_factor=0.25, mode='bilinear')
        mask_gt = mask_gt * border_mask

    # print(mask_gt.shape)
    # print(mask_gt.min(), mask_gt.max(), torch.sum(mask_gt))
    # print(flow_gt)

    if full_res:
        final_flow_scale = FLOW_SCALE * 4
        final_flow_fw = fw_flow[-1][0] * final_flow_scale
    else:
        final_flow_scale = FLOW_SCALE
        final_flow_fw = fw_flow[-1][0] * final_flow_scale * 4

    combined_loss = 0.0

    for i, flow in enumerate(flow_fw):
        loss = charbonnier_loss(flow * final_flow_scale - flow_gt, mask_gt)
        combined_loss += loss
        # print(i)

    if not return_flows:
        return combined_loss

    return combined_loss, final_flow_fw





