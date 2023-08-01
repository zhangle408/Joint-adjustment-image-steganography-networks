# encoding: utf-8

import functools

import torch
import torch.nn as nn
#from models.module import Harm2d
import torch.nn.functional as F
import math
import numpy as np


# Defines the Unet generator.
# |num_downs|: number of downsamplings in UNet. For example,
# if |num_downs| == 7, image of size 128x128 will become of size 1x1
# at the bottleneck

'''Hnet = UnetGenerator_C(input_nc=opt.channel_secret * opt.num_secret, output_nc=opt.channel_cover * opt.num_cover,
                             num_downs=num_downs, norm_layer=norm_layer, output_function=nn.Tanh)'''


class UnetGenerator_coarse(nn.Module):
    def __init__(self, input_nc, output_nc, num_downs, ngf=64,
                 norm_layer=None, use_dropout=False, use_tanh=1):
        super(UnetGenerator_coarse, self).__init__()
        self.use_tanh = use_tanh
        '''self.tanh = output_function==nn.Tanh
        if self.tanh:
            self.factor = 10/255
        else:
            self.factor = 1.0'''
        self.input_nc = input_nc
        self.factor = 10 / 255
        self.tanh = nn.Tanh
        self.conv1 = nn.Conv2d(input_nc, 64, kernel_size=4, stride=2, padding=1, bias=False)
        self.conv_1 = nn.Conv2d(input_nc, 64, kernel_size=1, stride=1, padding=0, bias=False)
        self.conv_2 = nn.Conv2d(input_nc, 128, kernel_size=1, stride=1, padding=0, bias=False)
        self.conv2 = nn.Conv2d(64*2, 128, kernel_size=4, stride=2, padding=1, bias=False)
        self.conv3 = nn.Conv2d(128*2, 256, kernel_size=4, stride=2, padding=1, bias=False)
        self.conv4 = nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1, bias=False)
        self.conv5 = nn.Conv2d(512, 512, kernel_size=4, stride=2, padding=1, bias=False)
        self.bn2 = norm_layer(128)
        self.bn3 = norm_layer(256)
        self.bn4 = norm_layer(512)
        self.leakyrelu = nn.LeakyReLU(0.2, True)
        self.relu = nn.ReLU()
        self.convtran5 = nn.ConvTranspose2d(512, 512, kernel_size=4, stride=2, padding=1, bias=False)
        self.bnt5 = norm_layer(512)
        self.convtran4 = nn.ConvTranspose2d(1024, 256, kernel_size=4, stride=2, padding=1, bias=False)
        self.bnt4 = norm_layer(256)
        self.convtran3 = nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1, bias=False)
        self.bnt3 = norm_layer(128)
        self.convtran2 = nn.ConvTranspose2d(256, 64, kernel_size=4, stride=2, padding=1, bias=False)
        self.bnt2 = norm_layer(64)
        self.convtran1 = nn.ConvTranspose2d(128, output_nc, kernel_size=4, stride=2, padding=1, bias=False)
        self.sigmoid = nn.Sigmoid()

        self.drop = nn.Dropout(0.5)

    def forward(self, input1, input_2, input_3):
        input_2 = self.conv_1(input_2)#b,64,64,64
        input_3 = self.conv_2(input_3)#b,128,32,32
        out_conv_1 = self.conv1(input1)#b,64,64,64
        Out1 = torch.cat([out_conv_1,input_2],dim=1)
        out_conv_2 = self.bn2(self.conv2(self.leakyrelu(Out1)))#b,128,32,32
        Out2 = torch.cat([out_conv_2, input_3], dim=1)
        out3 = self.bn3(self.conv3(self.leakyrelu(Out2)))#b,256,16,16
        out_3 = self.bnt3(self.convtran3(self.relu(out3)))#b,128,32,32
        #print(out_3.shape)
        #print('out_conv_2.shape', out_conv_2.shape)
        oout_3 = out_3
        out_3 = torch.cat([out_conv_2, out_3], 1)
        out_2 = self.bnt2(self.convtran2(self.relu(out_3)))
        oout_2 = out_2
        out_2 = torch.cat([out_conv_1, out_2], 1)
        out_1 = self.relu(out_2)
        out = self.convtran1(out_1)
        if self.use_tanh == 1:
            out = torch.tanh(out)
            out = self.factor * out
            #print('+++++')
        else:
            out = self.sigmoid(out)
            #print('-----')
        # out = torch.cat([input, out], 1)

        return out_conv_1, out_conv_2, out3, oout_3, oout_2, out


