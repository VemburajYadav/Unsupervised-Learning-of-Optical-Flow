import torch
import torch.nn as nn
import torch.nn.functional as F
from src.core.custom_modules.correlation import ModuleCorrelation
from src.core.image_warp import image_warp

FLOW_SCALE = 5.0

class Upconv(nn.Module):

    def __init__(self, bilinear_upsampling=True):
        super(Upconv, self).__init__()

        self.moduleSixOut = torch.nn.Conv2d(in_channels=1024, out_channels=2, kernel_size=3, stride=1,
                                            padding=1)

        self.moduleSixUp = nn.ConvTranspose2d(in_channels=2, out_channels=2, kernel_size=4, stride=2, padding=1)

        self.moduleFivNext = nn.Sequential(
            nn.ConvTranspose2d(in_channels=1024, out_channels=512, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(negative_slope=0.1, inplace=False))

        self.moduleFivOut = nn.Conv2d(in_channels=1026, out_channels=2, kernel_size=3, stride=1, padding=1)

        self.moduleFivUp = nn.ConvTranspose2d(in_channels=2, out_channels=2, kernel_size=4, stride=2,
                                               padding=1)

        self.moduleFouNext = nn.Sequential(
            nn.ConvTranspose2d(in_channels=1026, out_channels=256, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(negative_slope=0.1, inplace=False))

        self.moduleFouOut = nn.Conv2d(in_channels=770, out_channels=2, kernel_size=3, stride=1, padding=1)

        self.moduleFouUp = nn.ConvTranspose2d(in_channels=2, out_channels=2, kernel_size=4, stride=2, padding=1)

        self.moduleThrNext = nn.Sequential(
            nn.ConvTranspose2d(in_channels=770, out_channels=128, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(negative_slope=0.1, inplace=False))

        self.moduleThrOut = nn.Conv2d(in_channels=386, out_channels=2, kernel_size=3, stride=1, padding=1)

        self.moduleThrUp = nn.ConvTranspose2d(in_channels=2, out_channels=2, kernel_size=4, stride=2, padding=1)

        self.moduleTwoNext = nn.Sequential(
            nn.ConvTranspose2d(in_channels=386, out_channels=64, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(negative_slope=0.1, inplace=False))

        self.moduleTwoOut = nn.Conv2d(in_channels=194, out_channels=2, kernel_size=3, stride=1, padding=1)

        # self.moduleTwoUp = nn.ConvTranspose2d(in_channels=2, out_channels=2, kernel_size=4, stride=2, padding=1)
        #
        # self.moduleOneNext = nn.Sequential(
        #             nn.ConvTranspose2d(in_channels=194, out_channels=32, kernel_size=4, stride=2, padding=1),
        #             nn.LeakyReLU(negative_slope=0.1, inplace=False))
        #
        # self.moduleOneOut = nn.Conv2d(in_channels=98, out_channels=2, kernel_size=3, stride=1, padding=1)
        #
        # self.moduleOneUp = nn.ConvTranspose2d(in_channels=2, out_channels=2, kernel_size=4, stride=2, padding=1)
        #
        # self.moduleZeroNext = nn.Sequential(
        #     nn.ConvTranspose2d(in_channels=98, out_channels=16, kernel_size=4, stride=2, padding=1),
        #     nn.LeakyReLU(negative_slope=0.1, inplace=False))
        #
        # self.moduleZeroOut = nn.Conv2d(in_channels=50, out_channels=2, kernel_size=3, stride=1, padding=1)

        if bilinear_upsampling:
            self.moduleUpscale = nn.Upsample(scale_factor=2.0, mode='bilinear')
        else:
            self.moduleUpscale = nn.Sequential(
                nn.ConvTranspose2d(in_channels=2, out_channels=2, kernel_size=3, stride=2, padding=1, bias=False),
                nn.ReplicationPad2d(padding=[0, 1, 0, 1]))

    def forward(self, tensorFirst, tensorSecond, Conv_6, Conv_5, Conv_4, Conv_3, Conv_2):
        Flow_6 = self.moduleSixOut(Conv_6)

        tensorInput = torch.cat((Conv_5, self.moduleFivNext(Conv_6), self.moduleSixUp(Flow_6)), 1)
        Flow_5 = self.moduleFivOut(tensorInput)
        tensorInput = torch.cat((Conv_4, self.moduleFouNext(tensorInput), self.moduleFivUp(Flow_5)), 1)
        Flow_4 = self.moduleFouOut(tensorInput)
        tensorInput = torch.cat((Conv_3, self.moduleThrNext(tensorInput), self.moduleFouUp(Flow_4)), 1)
        Flow_3 = self.moduleThrOut(tensorInput)
        tensorInput = torch.cat((Conv_2, self.moduleTwoNext(tensorInput), self.moduleThrUp(Flow_3)), 1)
        Flow_2 = self.moduleTwoOut(tensorInput)

        Flow_1 = self.moduleUpscale(Flow_2)
        Flow_0 = self.moduleUpscale(Flow_1)

        return Flow_0, Flow_1, Flow_2, Flow_3, Flow_4, Flow_5, Flow_6


class Complex(nn.Module):

    def __init__(self, bilinear_upsampling=True):
        super(Complex, self).__init__()

        self.moduleOne = nn.Sequential(nn.ZeroPad2d([2, 4, 2, 4]),
                                       nn.Conv2d(in_channels=3, out_channels=64, kernel_size=7, stride=2,
                                                 padding=0),
                                       nn.LeakyReLU(negative_slope=0.1, inplace=False))

        self.moduleTwo = nn.Sequential(nn.ZeroPad2d([1, 3, 1, 3]),
                                       nn.Conv2d(in_channels=64, out_channels=128, kernel_size=5, stride=2,
                                                 padding=0),
                                       nn.LeakyReLU(negative_slope=0.1, inplace=False))

        self.moduleThr = nn.Sequential(nn.ZeroPad2d([1, 3, 1, 3]),
                                       nn.Conv2d(in_channels=128, out_channels=256, kernel_size=5, stride=2,
                                                 padding=0),
                                       nn.LeakyReLU(negative_slope=0.1, inplace=False))

        self.moduleRedir = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=32, kernel_size=1, stride=1, padding=0),
            nn.LeakyReLU(negative_slope=0.1, inplace=False))

        self.moduleCorrelation = ModuleCorrelation()

        self.moduleCombined = nn.Sequential(
            nn.Conv2d(in_channels=473, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(negative_slope=0.1, inplace=True))

        self.moduleFou = nn.Sequential(nn.ZeroPad2d([0, 2, 0, 2]),
                                       nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=2,
                                                 padding=0),
                                       nn.LeakyReLU(inplace=False, negative_slope=0.1),
                                       nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1,
                                                 padding=1),
                                       nn.LeakyReLU(inplace=False, negative_slope=0.1))

        self.moduleFiv = nn.Sequential(nn.ZeroPad2d([0, 2, 0, 2]),
                                       nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=2,
                                                  padding=0),
                                       nn.LeakyReLU(inplace=False, negative_slope=0.1),
                                       nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1,
                                                  padding=1),
                                       nn.LeakyReLU(inplace=False, negative_slope=0.1))

        self.moduleSix = nn.Sequential(nn.ZeroPad2d([0, 2, 0, 2]),
                                       nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=3, stride=2,
                                                 padding=0),
                                       nn.LeakyReLU(inplace=False, negative_slope=0.1),
                                       nn.Conv2d(in_channels=1024, out_channels=1024, kernel_size=3, stride=1,
                                                 padding=1),
                                       nn.LeakyReLU(inplace=False, negative_slope=0.1))

        self.moduleUpconv = Upconv(bilinear_upsampling=bilinear_upsampling)

    def forward(self, tensorFirst, tensorSecond, tensorFlow):
        assert (tensorFlow is None)

        Conv_1 = self.moduleOne(tensorFirst)
        Conv_2 = self.moduleTwo(Conv_1)
        Conv_3 = self.moduleThr(Conv_2)

        tensorOther = self.moduleThr(self.moduleTwo(self.moduleOne(tensorSecond)))
        tensorCorrelation = self.moduleCorrelation(Conv_3, tensorOther)

        tensorRedir = self.moduleRedir(Conv_3)
        Conv_3 = self.moduleCombined(torch.cat((tensorRedir, tensorCorrelation), 1))
        Conv_4 = self.moduleFou(Conv_3)
        Conv_5 = self.moduleFiv(Conv_4)
        Conv_6 = self.moduleSix(Conv_5)


        return self.moduleUpconv(tensorFirst, tensorSecond, Conv_6, Conv_5, Conv_4, Conv_3, Conv_2)


class Simple(nn.Module):

    def __init__(self, bilinear_upsampling=True):
        super(Simple, self).__init__()

        self.moduleOne = nn.Sequential(nn.ZeroPad2d([2, 4, 2, 4]),
                                       nn.Conv2d(in_channels=14, out_channels=64, kernel_size=7, stride=2,
                                                 padding=0),
                                       nn.LeakyReLU(negative_slope=0.1, inplace=False))

        self.moduleTwo = nn.Sequential(nn.ZeroPad2d([1, 3, 1, 3]),
                                       nn.Conv2d(in_channels=64, out_channels=128, kernel_size=5, stride=2,
                                                 padding=0),
                                       nn.LeakyReLU(negative_slope=0.1, inplace=False))

        self.moduleThr = nn.Sequential(nn.ZeroPad2d([1, 3, 1, 3]),
                                       nn.Conv2d(in_channels=128, out_channels=256, kernel_size=5, stride=2,
                                                 padding=0),
                                       nn.LeakyReLU(negative_slope=0.1, inplace=False),
                                       nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1,
                                                 padding=1),
                                       nn.LeakyReLU(negative_slope=0.1, inplace=False))

        self.moduleFou = nn.Sequential(nn.ZeroPad2d([0, 2, 0, 2]),
                                       nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=2,
                                                 padding=0),
                                       nn.LeakyReLU(inplace=False, negative_slope=0.1),
                                       nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1,
                                                 padding=1),
                                       nn.LeakyReLU(inplace=False, negative_slope=0.1))

        self.moduleFiv = nn.Sequential(nn.ZeroPad2d([0, 2, 0, 2]),
                                        nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=2,
                                                  padding=0),
                                        nn.LeakyReLU(inplace=False, negative_slope=0.1),
                                        nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1,
                                                  padding=1),
                                        nn.LeakyReLU(inplace=False, negative_slope=0.1))

        self.moduleSix = nn.Sequential(nn.ZeroPad2d([0, 2, 0, 2]),
                                       nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=3, stride=2,
                                                 padding=0),
                                       nn.LeakyReLU(inplace=False, negative_slope=0.1),
                                       nn.Conv2d(in_channels=1024, out_channels=1024, kernel_size=3, stride=1,
                                                 padding=1),
                                       nn.LeakyReLU(inplace=False, negative_slope=0.1))

        self.moduleUpconv = Upconv(bilinear_upsampling=bilinear_upsampling)

    def forward(self, tensorFirst, tensorSecond, tensorFlow):

        tensorWarp = image_warp(tensorSecond, tensorFlow)

        Conv_1 = self.moduleOne(torch.cat([tensorFirst, tensorSecond, tensorFlow,
                                           tensorWarp, (tensorFirst - tensorWarp).abs()], dim=1))
        Conv_2 = self.moduleTwo(Conv_1)
        Conv_3 = self.moduleThr(Conv_2)
        Conv_4 = self.moduleFou(Conv_3)
        Conv_5 = self.moduleFiv(Conv_4)
        Conv_6 = self.moduleSix(Conv_5)


        return self.moduleUpconv(tensorFirst, tensorSecond, Conv_6, Conv_5, Conv_4, Conv_3, Conv_2)



class Network(nn.Module):

    def __init__(self):
        super(Network, self).__init__()

        self.moduleFlownets = torch.nn.ModuleList([
            Complex(),
            Simple(),
            Simple()
        ])

        self.load_state_dict(torch.load('../../network-css.pytorch'))


    def forward(self, tensorFirst, tensorSecond):

        tensorFirst = tensorFirst[:, [2, 1, 0], :, :]
        tensorSecond = tensorSecond[:, [2, 1, 0], :, :]

        tensorFirst[:, 0, :, :] = tensorFirst[:, 0, :, :] - (104.920005 / 255.0)
        tensorFirst[:, 1, :, :] = tensorFirst[:, 1, :, :] - (110.175300 / 255.0)
        tensorFirst[:, 2, :, :] = tensorFirst[:, 2, :, :] - (114.785955 / 255.0)

        tensorSecond[:, 0, :, :] = tensorSecond[:, 0, :, :] - (104.920005 / 255.0)
        tensorSecond[:, 1, :, :] = tensorSecond[:, 1, :, :] - (110.175300 / 255.0)
        tensorSecond[:, 2, :, :] = tensorSecond[:, 2, :, :] - (114.785955 / 255.0)

        tensorFlow = []

        tensorFlow.append(self.moduleFlownets[0](tensorFirst, tensorSecond, None))
        tensorFlow.append(self.moduleFlownets[1](tensorFirst, tensorSecond, tensorFlow[-1][-1]))
        tensorFlow.append(self.moduleFlownets[2](tensorFirst, tensorSecond, tensorFlow[-1][-1]))

        return tensorFlow












