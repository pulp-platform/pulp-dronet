# Copyright (C) 2020 ETH Zurich, Switzerland
# All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# See LICENSE.apache.md in the top directory for details.
# You may obtain a copy of the License at

    # http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# File:    dronet_v2.py
# Original author:  Daniele Palossi <dpalossi@iis.ee.ethz.ch>
# Author:           Lorenzo Lamberti <lorenzo.lamberti@unibo.it>
# Date:    5.1.2021

import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim
import nemo

################################################################################
# PULP-Dronet building blocks #
################################################################################
class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=2):
        super(ResBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=stride, padding=1, dilation=1, groups=1, bias=False, padding_mode='zeros')
        self.conv2 = nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1, dilation=1, groups=1, bias=False, padding_mode='zeros')
        self.bn1 = nn.BatchNorm2d(num_features=out_channels, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.bn2 = nn.BatchNorm2d(num_features=out_channels, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu1 = nn.ReLU6(inplace=False)
        self.relu2 = nn.ReLU6(inplace=False)
        self.bypass = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=stride, padding=0, dilation=1, groups=1, bias=False, padding_mode='zeros')
        self.bn_bypass = nn.BatchNorm2d(num_features=out_channels, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu3 = nn.ReLU6(inplace=False)
        self.add = nemo.quant.pact.PACT_IntegerAdd()

    def forward(self, x):
        identity = x
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        x_bypass = self.bypass(identity)
        x_bypass = self.bn_bypass(x_bypass)
        x_bypass = self.relu3(x_bypass)
        x = self.add(x, x_bypass)
        return x


class Depthwise_Separable(nn.Module):
    """ Block involving a deptwise convolution followed by a pointwise convolution.
        The first one applies a channel-wise KxK convolution, while the second one
        is the 1x1 convolution that changes the channel dimension. The resulting
        computation reduction is of 1/OutChannels + 1/K^2"""

    def __init__(self, in_channels, out_channels):
        super(Depthwise_Separable, self).__init__()
        self.depthconv1 = nn.Conv2d(in_channels=in_channels, out_channels=in_channels,
                                    kernel_size=3, stride=2, padding=1, dilation=1,
                                    groups=in_channels, bias=False, padding_mode='zeros')
        self.bn1 = nn.BatchNorm2d(num_features=in_channels, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu1 = nn.ReLU6(inplace=False)

        self.pointwise1 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm2d(num_features=out_channels, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu2 = nn.ReLU6(inplace=False)

        self.depthconv2 = nn.Conv2d(in_channels=out_channels, out_channels=out_channels,
                                    kernel_size=3, stride=1, padding=1, dilation=1,
                                    groups=out_channels, bias=False, padding_mode='zeros')
        self.bn3 = nn.BatchNorm2d(num_features=out_channels, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu3 = nn.ReLU6(inplace=False)

        self.pointwise2 = nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=1, bias=False)
        self.bn4 = nn.BatchNorm2d(num_features=out_channels, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu4 = nn.ReLU6(inplace=False)

        self.bypass_conv= nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                                    kernel_size=1, stride=2, padding=0, groups=1, bias=False)
        self.bypass_bn = nn.BatchNorm2d(num_features=out_channels)
        self.relu_bypass = nn.ReLU6(inplace=False)
        self.add = nemo.quant.pact.PACT_IntegerAdd()

    def forward(self, x):
        identity = x
        out = self.depthconv1(x)
        out = self.bn1(out)
        out = self.relu1(out)

        out = self.pointwise1(out)
        out = self.bn2(out)
        out = self.relu2(out)

        out = self.depthconv2(out)
        out = self.bn3(out)
        out = self.relu3(out)

        out = self.pointwise2(out)
        out = self.bn4(out)
        out = self.relu4(out)

        bypass = self.bypass_conv(identity)
        bypass = self.bypass_bn(bypass)
        bypass = self.relu_bypass(bypass)

        out = self.add(out, bypass)

        return out


class Inverted_Linear_Bottleneck(nn.Module):
    """ Block involving an inverted connection following a narrow-wide-narrow approach
        in terms of channels. The first step widens the network by an 'expand' factor.
        The following depthwise convolution reduces the number of learnable parameters,
        while the last 1x1 convolution squeezes the network. The last layer does not
        present any ReLU activation (linear output) for channel dimension motivations."""
    def __init__(self, in_channels, out_channels, expand):
        super(Inverted_Linear_Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=in_channels*expand, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(num_features=in_channels*expand)
        self.relu1 = nn.ReLU6(inplace=False)

        self.depthconv = nn.Conv2d(in_channels=in_channels*expand, out_channels=in_channels*expand, kernel_size=3,
                                   stride=2, padding=1, bias=False, groups=in_channels*expand, padding_mode='zeros')
        self.bn2 = nn.BatchNorm2d(num_features=in_channels*expand)
        self.relu2 = nn.ReLU6(inplace=False)


        self.conv2 = nn.Conv2d(in_channels=in_channels*expand, out_channels=out_channels, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(num_features=out_channels)
        self.relu3 = nn.ReLU6(inplace=False)

        self.bypass_conv= nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                            kernel_size=1, stride=2, padding=0, groups=1, bias=False)
        self.bypass_bn = nn.BatchNorm2d(num_features=out_channels)
        self.add = nemo.quant.pact.PACT_IntegerAdd()


    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)

        out = self.depthconv(out)
        out = self.bn2(out)
        out = self.relu2(out)

        out = self.conv2(out)
        out = self.bn3(out)

        bypass = self.bypass_conv(identity)
        bypass = self.bypass_bn(bypass)
        bypass = self.relu3(bypass)

        out = self.add(out, bypass)

        return out

################################################################################
# PULP-Dronet CNN #
################################################################################
class dronet(nn.Module):
    ResBlock = ResBlock
    Depthwise_Separable = Depthwise_Separable
    def __init__(self, depth_mult=1.0, block_class=ResBlock, nemo=False):
        super(dronet, self).__init__()
        self.nemo=nemo # Prepare network for quantization? [True, False]
        first_conv_channels=int(32*depth_mult)
        #conv 5x5, 1, 32, 200x200, /2
        self.first_conv = nn.Conv2d(in_channels=1, out_channels=first_conv_channels, kernel_size=5, stride=2, padding=2, dilation=1, groups=1, bias=False, padding_mode='zeros')
        self.bn1 = nn.BatchNorm2d(num_features=first_conv_channels, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu1 = nn.ReLU6(inplace=False)
        #max pooling 2x2, 32, 32, 100x100, /2
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, return_indices=False, ceil_mode=False)
        self.resBlock1 = block_class(first_conv_channels, first_conv_channels)
        self.resBlock2 = block_class(first_conv_channels, first_conv_channels*2)
        self.resBlock3 = block_class(first_conv_channels*2, first_conv_channels*4)
        if not self.nemo: self.dropout = nn.Dropout(p=0.5, inplace=False)
        fc_size = (first_conv_channels*4)*7*7
        self.fc = nn.Linear(in_features=fc_size, out_features=2, bias=False)
        self.sig = nn.Sigmoid()

    def forward(self, x):
        x = self.first_conv(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.pool(x)
        x = self.resBlock1(x)
        x = self.resBlock2(x)
        x = self.resBlock3(x)
        if not self.nemo: x = self.dropout(x)
        x = x.flatten(1)
        x = self.fc(x)
        steer = x[:, 0]
        coll = self.sig(x[:, 1])
        return [steer, coll]


################################################################################
# Classification only #
################################################################################

class dronet_classification(nn.Module):
    def __init__(self, depth_mult=1.0, block_class=ResBlock, yawrate_classes=3, nemo=False):
        super(dronet_classification, self).__init__()
        self.nemo=nemo # Prepare network for quantization? [True, False]
        first_conv_channels=int(32*depth_mult)
        #conv 5x5, 1, 32, 200x200, /2
        self.first_conv = nn.Conv2d(in_channels=1, out_channels=first_conv_channels, kernel_size=5, stride=2, padding=2, dilation=1, groups=1, bias=False, padding_mode='zeros')
        self.bn1 = nn.BatchNorm2d(num_features=first_conv_channels, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu1 = nn.ReLU6(inplace=False) # chek with 3 different
        #max pooling 2x2, 32, 32, 100x100, /2
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, return_indices=False, ceil_mode=False)
        self.resBlock1 = block_class(first_conv_channels, first_conv_channels)
        self.resBlock2 = block_class(first_conv_channels, first_conv_channels*2)
        self.resBlock3 = block_class(first_conv_channels*2, first_conv_channels*4)
        if not self.nemo: self.dropout = nn.Dropout(p=0.5, inplace=False)

        fc_size = (first_conv_channels*4)*7*7
        self.fc = nn.Linear(in_features=fc_size, out_features=1+yawrate_classes, bias=False)
        self.sig = nn.Sigmoid()
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, x):
        x = self.first_conv(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.pool(x)
        x = self.resBlock1(x)
        x = self.resBlock2(x)
        x = self.resBlock3(x)
        if not self.nemo: x = self.dropout(x)
        x = x.flatten(1)
        x = self.fc(x)
        steer = x[:, [0,1,2]]
        steer = self.softmax(steer)
        coll = self.sig(x[:, 3])
        return [steer, coll]



################################################################################
# TOPOLOGY MODIFICATIONS #
################################################################################

class dronet_x2_resblocks(nn.Module):
    def __init__(self, depth_mult=1.0):
        super(dronet_x2_resblocks, self).__init__()
        first_conv_channels=int(32*depth_mult)
        #conv 5x5, 1, 32, 200x200, /2
        self.first_conv = nn.Conv2d(in_channels=1, out_channels=first_conv_channels, kernel_size=5, stride=2, padding=2, dilation=1, groups=1, bias=False, padding_mode='zeros')
        self.bn1 = nn.BatchNorm2d(num_features=first_conv_channels, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu1 = nn.ReLU6(inplace=False)
        #max pooling 2x2, 32, 32, 100x100, /2
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, return_indices=False, ceil_mode=False)
        self.resBlock1a = ResBlock(first_conv_channels, first_conv_channels)
        self.resBlock1b = ResBlock(first_conv_channels, first_conv_channels, stride=1)
        self.resBlock2a = ResBlock(first_conv_channels, first_conv_channels*2)
        self.resBlock2b = ResBlock(first_conv_channels*2, first_conv_channels*2, stride=1)
        self.resBlock3a = ResBlock(first_conv_channels*2, first_conv_channels*4)
        self.resBlock3b = ResBlock(first_conv_channels*4, first_conv_channels*4, stride=1)
        self.dropout = nn.Dropout(p=0.5, inplace=False)

        fc_size = (first_conv_channels*4)*7*7
        self.fc = nn.Linear(in_features=fc_size, out_features=2, bias=False)
        self.sig = nn.Sigmoid()

    def forward(self, x):
        x = self.first_conv(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.pool(x)
        x = self.resBlock1a(x)
        x = self.resBlock1b(x)
        x = self.resBlock2a(x)
        x = self.resBlock2b(x)
        x = self.resBlock3a(x)
        x = self.resBlock3b(x)
        x = self.dropout(x)
        x = x.flatten(1)
        x = self.fc(x)
        steer = x[:, 0]
        coll = self.sig(x[:, 1])
        return [steer, coll]




class dronet_x3_resblocks(nn.Module):
    def __init__(self, depth_mult=1.0):
        super(dronet_x3_resblocks, self).__init__()
        first_conv_channels=int(32*depth_mult)
        #conv 5x5, 1, 32, 200x200, /2
        self.first_conv = nn.Conv2d(in_channels=1, out_channels=first_conv_channels, kernel_size=5, stride=2, padding=2, dilation=1, groups=1, bias=False, padding_mode='zeros')
        self.bn1 = nn.BatchNorm2d(num_features=first_conv_channels, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu1 = nn.ReLU6(inplace=False)
        #max pooling 2x2, 32, 32, 100x100, /2
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, return_indices=False, ceil_mode=False)
        self.resBlock1a = ResBlock(first_conv_channels, first_conv_channels)
        self.resBlock1b = ResBlock(first_conv_channels, first_conv_channels, stride=1)
        self.resBlock1c = ResBlock(first_conv_channels, first_conv_channels, stride=1)
        self.resBlock2a = ResBlock(first_conv_channels, first_conv_channels*2)
        self.resBlock2b = ResBlock(first_conv_channels*2, first_conv_channels*2, stride=1)
        self.resBlock2c = ResBlock(first_conv_channels*2, first_conv_channels*2, stride=1)
        self.resBlock3a = ResBlock(first_conv_channels*2, first_conv_channels*4)
        self.resBlock3b = ResBlock(first_conv_channels*4, first_conv_channels*4, stride=1)
        self.resBlock3c = ResBlock(first_conv_channels*4, first_conv_channels*4, stride=1)
        self.dropout = nn.Dropout(p=0.5, inplace=False)

        fc_size = (first_conv_channels*4)*7*7
        self.fc = nn.Linear(in_features=fc_size, out_features=2, bias=False)
        self.sig = nn.Sigmoid()

    def forward(self, x):
        x = self.first_conv(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.pool(x)
        x = self.resBlock1a(x)
        x = self.resBlock1b(x)
        x = self.resBlock1c(x)
        x = self.resBlock2a(x)
        x = self.resBlock2b(x)
        x = self.resBlock2c(x)
        x = self.resBlock3a(x)
        x = self.resBlock3b(x)
        x = self.resBlock3c(x)
        x = self.dropout(x)
        x = x.flatten(1)
        x = self.fc(x)
        steer = x[:, 0]
        coll = self.sig(x[:, 1])
        return [steer, coll]
