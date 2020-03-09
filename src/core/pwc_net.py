import torch
import torch.nn as nn
import torch.nn.functional as F
from src.core.custom_modules.correlation import ModuleCorrelation
from src.core.image_warp import Warp
import numpy as np


FLOW_SCALE = 5.0

class FeatureExtractor(nn.Module):

    def __init__(self):
        super(FeatureExtractor, self).__init__()

        self.moduleOneOne = nn.Sequential(nn.ZeroPad2d([0, 2, 0, 2]),
                                          nn.Conv2d(in_channels=3, out_channels=16,
                                                    kernel_size=3, stride=2, padding=0),
                                          nn.LeakyReLU(negative_slope=0.1, inplace=False))

        self.moduleOneTwo = nn.Sequential(nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3,
                                                    stride=1, padding=1),
                                          nn.LeakyReLU(negative_slope=0.1, inplace=False))

        self.moduleTwoOne = nn.Sequential(nn.ZeroPad2d([0, 2, 0, 2]),
                                          nn.Conv2d(in_channels=16, out_channels=32,
                                                    kernel_size=3, stride=2, padding=0),
                                          nn.LeakyReLU(negative_slope=0.1, inplace=False))

        self.moduleTwoTwo = nn.Sequential(nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3,
                                                    stride=1, padding=1),
                                          nn.LeakyReLU(negative_slope=0.1, inplace=False))

        self.moduleThreeOne = nn.Sequential(nn.ZeroPad2d([0, 2, 0, 2]),
                                            nn.Conv2d(in_channels=32, out_channels=64,
                                                      kernel_size=3, stride=2, padding=0),
                                            nn.LeakyReLU(negative_slope=0.1, inplace=False))

        self.moduleThreeTwo = nn.Sequential(nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3,
                                                      stride=1, padding=1),
                                            nn.LeakyReLU(negative_slope=0.1, inplace=False))

        self.moduleFourOne = nn.Sequential(nn.ZeroPad2d([0, 2, 0, 2]),
                                           nn.Conv2d(in_channels=64, out_channels=96,
                                                     kernel_size=3, stride=2, padding=0),
                                           nn.LeakyReLU(negative_slope=0.1, inplace=False))

        self.moduleFourTwo = nn.Sequential(nn.Conv2d(in_channels=96, out_channels=96, kernel_size=3,
                                                     stride=1, padding=1),
                                           nn.LeakyReLU(negative_slope=0.1, inplace=False))

        self.moduleFiveOne = nn.Sequential(nn.ZeroPad2d([0, 2, 0, 2]),
                                           nn.Conv2d(in_channels=96, out_channels=128,
                                                     kernel_size=3, stride=2, padding=0),
                                           nn.LeakyReLU(negative_slope=0.1, inplace=False))

        self.moduleFiveTwo = nn.Sequential(nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3,
                                                     stride=1, padding=1),
                                           nn.LeakyReLU(negative_slope=0.1, inplace=False))

        self.moduleSixOne = nn.Sequential(nn.ZeroPad2d([0, 2, 0, 2]),
                                          nn.Conv2d(in_channels=128, out_channels=196,
                                                    kernel_size=3, stride=2, padding=0),
                                          nn.LeakyReLU(negative_slope=0.1, inplace=False))

        self.moduleSixTwo = nn.Sequential(nn.Conv2d(in_channels=196, out_channels=196, kernel_size=3,
                                                    stride=1, padding=1),
                                          nn.LeakyReLU(negative_slope=0.1, inplace=False))

    def forward(self, x):

        Conv1_1 = self.moduleOneOne(x)
        Conv1_2 = self.moduleOneTwo(Conv1_1)

        Conv2_1 = self.moduleTwoOne(Conv1_2)
        Conv2_2 = self.moduleTwoTwo(Conv2_1)

        Conv3_1 = self.moduleThreeOne(Conv2_2)
        Conv3_2 = self.moduleThreeTwo(Conv3_1)

        Conv4_1 = self.moduleFourOne(Conv3_2)
        Conv4_2 = self.moduleFourTwo(Conv4_1)

        Conv5_1 = self.moduleFiveOne(Conv4_2)
        Conv5_2 = self.moduleFiveTwo(Conv5_1)

        Conv6_1 = self.moduleSixOne(Conv5_2)
        Conv6_2 = self.moduleSixTwo(Conv6_1)

        return Conv1_2, Conv2_2, Conv3_2, Conv4_2, Conv5_2, Conv6_2


class CostVolumeNetwork(nn.Module):

    def __init__(self, pad_size=4, max_displacement=4, stride_1=1, stride_2=1, kernel_size=1):
        super(CostVolumeNetwork, self).__init__()

        self.moduleCorr = ModuleCorrelation(pad_size=pad_size,
                                            max_displacement=max_displacement,
                                            stride_1=stride_1,
                                            stride_2=stride_2,
                                            kernel_size=kernel_size)
        self.leakyReLU = nn.LeakyReLU(negative_slope=0.1, inplace=False)

    def forward(self, x1, x2):

        cost_volume = self.leakyReLU(self.moduleCorr(x1, x2))

        return cost_volume


class EstimatorNetwork(nn.Module):

    def __init__(self, in_channels=None):
        super(EstimatorNetwork, self).__init__()

        channels = np.cumsum([128, 128, 96, 64, 32])

        self.Conv1 = nn.Sequential(nn.Conv2d(in_channels=in_channels, out_channels=128,
                                   kernel_size=3, stride=1, padding=1),
                                   nn.LeakyReLU(negative_slope=0.1, inplace=False))

        self.Conv2 = nn.Sequential(nn.Conv2d(in_channels=in_channels+channels[0], out_channels=128,
                                   kernel_size=3, stride=1, padding=1),
                                   nn.LeakyReLU(negative_slope=0.1, inplace=False))

        self.Conv3 = nn.Sequential(nn.Conv2d(in_channels=in_channels+channels[1], out_channels=96,
                                   kernel_size=3, stride=1, padding=1),
                                   nn.LeakyReLU(negative_slope=0.1, inplace=False))

        self.Conv4 = nn.Sequential(nn.Conv2d(in_channels=in_channels+channels[2], out_channels=64,
                                   kernel_size=3, stride=1, padding=1),
                                   nn.LeakyReLU(negative_slope=0.1, inplace=False))

        self.Conv5 = nn.Sequential(nn.Conv2d(in_channels=in_channels+channels[3], out_channels=32,
                                   kernel_size=3, stride=1, padding=1),
                                   nn.LeakyReLU(negative_slope=0.1, inplace=False))

        self.Flow = nn.Sequential(nn.Conv2d(in_channels=in_channels+channels[4], out_channels=2,
                                  kernel_size=3, stride=1, padding=1),
                                  nn.LeakyReLU(negative_slope=0.1, inplace=False))

        self.UpFeat = nn.Sequential(
            nn.ConvTranspose2d(in_channels=in_channels+channels[4], out_channels=2, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(negative_slope=0.1, inplace=False))

        self.UpFlow = nn.Sequential(
            nn.ConvTranspose2d(in_channels=2, out_channels=2, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(negative_slope=0.1, inplace=False))

    def forward(self, x):

        net_conv1 = torch.cat((self.Conv1(x), x), dim=1)
        net_conv2 = torch.cat((self.Conv2(net_conv1), net_conv1), dim=1)
        net_conv3 = torch.cat((self.Conv3(net_conv2), net_conv2), dim=1)
        net_conv4 = torch.cat((self.Conv4(net_conv3), net_conv3), dim=1)
        net_conv5 = torch.cat((self.Conv5(net_conv4), net_conv4), dim=1)

        flow = self.Flow(net_conv5)
        up_flow = self.UpFlow(flow)
        up_feat = self.UpFeat(net_conv5)

        return flow, up_flow, up_feat, net_conv5


class ContextNetwork(nn.Module):

    def __init__(self, in_channels=68):
        super(ContextNetwork, self).__init__()

        self.DilatedConv1 = nn.Sequential(nn.Conv2d(in_channels=in_channels, out_channels=128,
                                          kernel_size=3, stride=1, padding=1),
                                          nn.LeakyReLU(negative_slope=0.1, inplace=False))

        self.DilatedConv2 = nn.Sequential(nn.Conv2d(in_channels=128, out_channels=128,
                                          kernel_size=3, stride=1, dilation=2, padding=2),
                                          nn.LeakyReLU(negative_slope=0.1, inplace=False))

        self.DilatedConv3 = nn.Sequential(nn.Conv2d(in_channels=128, out_channels=128,
                                          kernel_size=3, stride=1, dilation=4, padding=4),
                                          nn.LeakyReLU(negative_slope=0.1, inplace=False))

        self.DilatedConv4 = nn.Sequential(nn.Conv2d(in_channels=128, out_channels=96,
                                          kernel_size=3, stride=1, dilation=8, padding=8),
                                          nn.LeakyReLU(negative_slope=0.1, inplace=False))

        self.DilatedConv5 = nn.Sequential(nn.Conv2d(in_channels=96, out_channels=64,
                                          kernel_size=3, stride=1, dilation=16, padding=16),
                                          nn.LeakyReLU(negative_slope=0.1, inplace=False))

        self.DilatedConv6 = nn.Sequential(nn.Conv2d(in_channels=64, out_channels=32,
                                          kernel_size=3, stride=1, padding=1),
                                          nn.LeakyReLU(negative_slope=0.1, inplace=False))

        self.DilatedConv7 = nn.Sequential(nn.Conv2d(in_channels=32, out_channels=2,
                                          kernel_size=3, stride=1, padding=1),
                                          nn.LeakyReLU(negative_slope=0.1, inplace=False))

    def forward(self, x):

        net_dilated_conv1 = self.DilatedConv1(x)
        net_dilated_conv2 = self.DilatedConv2(net_dilated_conv1)
        net_dilated_conv3 = self.DilatedConv3(net_dilated_conv2)
        net_dilated_conv4 = self.DilatedConv4(net_dilated_conv3)
        net_dilated_conv5 = self.DilatedConv5(net_dilated_conv4)
        net_dilated_conv6 = self.DilatedConv6(net_dilated_conv5)
        net_dilated_conv7 = self.DilatedConv7(net_dilated_conv6)

        return net_dilated_conv7


class PWCNet(nn.Module):

    def __init__(self):
        super(PWCNet, self).__init__()

        self.feature_extractor = FeatureExtractor()
        self.cost_volume = CostVolumeNetwork(pad_size=4, max_displacement=4, stride_1=1, stride_2=1, kernel_size=1)

        self.warp = Warp()

        channels = np.array([128, 96, 64, 32]) + 4 + 81

        self.estimator_6 = EstimatorNetwork(in_channels=81)
        self.estimator_5 = EstimatorNetwork(in_channels=channels[0])
        self.estimator_4 = EstimatorNetwork(in_channels=channels[1])
        self.estimator_3 = EstimatorNetwork(in_channels=channels[2])
        self.estimator_2 = EstimatorNetwork(in_channels=channels[3])

        context_in_channels = channels[3] + 448

        self.context_network = ContextNetwork(in_channels=context_in_channels)
        self.moduleUpscale = nn.Upsample(scale_factor=2.0, mode='bilinear')

    def forward(self, tensorFirst, tensorSecond):

        x1_1, x1_2, x1_3, x1_4, x1_5, x1_6 = self.feature_extractor(tensorFirst)
        x2_1, x2_2, x2_3, x2_4, x2_5, x2_6 = self.feature_extractor(tensorSecond)

        corr6 = self.cost_volume(x1_6, x2_6)
        flow_6, up_flow_6, up_feat_6, context_6 = self.estimator_6(corr6)

        warp5 = self.warp(x2_5, up_flow_6 * 0.625)
        corr5 = self.cost_volume(x1_5, warp5)
        estimator_5_input = torch.cat((corr5, x1_5, up_flow_6, up_feat_6), dim=1)
        flow_5, up_flow_5, up_feat_5, context_5 = self.estimator_5(estimator_5_input)

        warp4 = self.warp(x2_4, up_flow_5 * 1.25)
        corr4 = self.cost_volume(x1_4, warp4)
        estimator_4_input = torch.cat((corr4, x1_4, up_flow_5, up_feat_5), dim=1)
        flow_4, up_flow_4, up_feat_4, context_4 = self.estimator_4(estimator_4_input)

        warp3 = self.warp(x2_3, up_flow_4 * 2.5)
        corr3 = self.cost_volume(x1_3, warp3)
        estimator_3_input = torch.cat((corr3, x1_3, up_flow_4, up_feat_4), dim=1)
        flow_3, up_flow_3, up_feat_3, context_3 = self.estimator_3(estimator_3_input)

        warp2 = self.warp(x2_2, up_flow_3 * 5.0)
        corr2 = self.cost_volume(x1_2, warp2)
        estimator_2_input = torch.cat((corr2, x1_2, up_flow_3, up_feat_3), dim=1)
        flow_2, up_flow_2, up_feat_2, context_2 = self.estimator_2(estimator_2_input)

        flow_2_context = self.context_network(context_2)

        flow_2_res = flow_2 + flow_2_context

        flow_1 = self.moduleUpscale(flow_2_res)
        flow_0 = self.moduleUpscale(flow_1)

        return flow_0, flow_1, flow_2_res, flow_3, flow_4, flow_5, flow_6






