#-----------------------------------------------------------------------------#
# Copyright(C) 2024 University of Bologna, Italy, ETH Zurich, Switzerland.    #
# All rights reserved.                                                        #
#                                                                             #
# Licensed under the Apache License, Version 2.0 (the "License");             #
# you may not use this file except in compliance with the License.            #
# See LICENSE in the top directory for details.                               #
# You may obtain a copy of the License at                                     #
#                                                                             #
#   http://www.apache.org/licenses/LICENSE-2.0                                #
#                                                                             #
# Unless required by applicable law or agreed to in writing, software         #
# distributed under the License is distributed on an "AS IS" BASIS,           #
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.    #
# See the License for the specific language governing permissions and         #
# limitations under the License.                                              #
#                                                                             #
# File:    dronet_v3.py                                                       #
# Authors:                                                                    #
#          Lorenzo Lamberti <lorenzo.lamberti@unibo.it>                       #
#          Lorenzo Bellone  <lorenzo.bellone@tii.ae>                          #
#          Daniele Palossi  <dpalossi@iis.ee.ethz.ch>                         #
# Date:    01.03.2024                                                         #
#-----------------------------------------------------------------------------#

import torch.nn as nn
import sys
sys.path.append('./nemo-dory/nemo')
try:
    import nemo
    print("nemo imported successfully")
except ModuleNotFoundError:
    print("Failed to import nemo")
    print(sys.path)


################################################################################
# PULP-Dronet building blocks #
################################################################################
class ResBlock(nn.Module):
    """
    Block involving two 3x3 convolutions with a ReLU activation function.
    The first convolution is applied with a stride of 2, while the second one
    is applied with a stride of 1. The bypass connection is a 1x1 convolution
    with a stride of 2. The output of the block is the sum of the two convolutions.
    """
    def __init__(self, in_channels, out_channels, stride=2, bypass=True):
        super(ResBlock, self).__init__()
        self.bypass = bypass
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=stride, padding=1, dilation=1, groups=1, bias=False, padding_mode='zeros')
        self.conv2 = nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1, dilation=1, groups=1, bias=False, padding_mode='zeros')
        self.bn1 = nn.BatchNorm2d(num_features=out_channels, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.bn2 = nn.BatchNorm2d(num_features=out_channels, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu1 = nn.ReLU6(inplace=False)
        self.relu2 = nn.ReLU6(inplace=False)
        if bypass:
            self.bypass = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=stride, padding=0, dilation=1, groups=1, bias=False, padding_mode='zeros')
            self.bn_bypass = nn.BatchNorm2d(num_features=out_channels, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            self.relu3 = nn.ReLU6(inplace=False)
            self.add = nemo.quant.pact.PACT_IntegerAdd()

    def forward(self, input):
        identity = input
        out = self.conv1(input)
        out = self.bn1(out)
        out = self.relu1(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu2(out)

        if self.bypass:
            out_bypass = self.bypass(identity)
            out_bypass = self.bn_bypass(out_bypass)
            out_bypass = self.relu3(out_bypass)
            out = self.add(out, out_bypass)
        return out


class Depthwise_Separable(nn.Module):
    """
    Block involving a deptwise convolution followed by a pointwise convolution.
    The first one applies a channel-wise KxK convolution, while the second one
    is the 1x1 convolution that changes the channel dimension. The resulting
    computation reduction is of 1/OutChannels + 1/K^2
    """
    def __init__(self, in_channels, out_channels, stride=2, bypass=True):
        super(Depthwise_Separable, self).__init__()
        self.bypass = bypass

        # Depthwise convolution
        self.depthconv1 = nn.Conv2d(in_channels=in_channels, out_channels=in_channels,
                                    kernel_size=3, stride=stride, padding=1, dilation=1,
                                    groups=in_channels, bias=False, padding_mode='zeros')
        self.bn1 = nn.BatchNorm2d(num_features=in_channels, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu1 = nn.ReLU6(inplace=False)
        # Pointwise convolution
        self.pointwise1 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm2d(num_features=out_channels, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu2 = nn.ReLU6(inplace=False)
        # Second Depthwise convolution
        self.depthconv2 = nn.Conv2d(in_channels=out_channels, out_channels=out_channels,
                                    kernel_size=3, stride=1, padding=1, dilation=1,
                                    groups=out_channels, bias=False, padding_mode='zeros')
        self.bn3 = nn.BatchNorm2d(num_features=out_channels, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu3 = nn.ReLU6(inplace=False)
        # Second Pointwise convolution
        self.pointwise2 = nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=1, bias=False)
        self.bn4 = nn.BatchNorm2d(num_features=out_channels, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu4 = nn.ReLU6(inplace=False)

        # Bypass connection
        if self.bypass:
            self.bypass_conv= nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                                        kernel_size=1, stride=stride, padding=0, groups=1, bias=False)
            self.bypass_bn = nn.BatchNorm2d(num_features=out_channels)
            self.relu_bypass = nn.ReLU6(inplace=False)
            self.add = nemo.quant.pact.PACT_IntegerAdd()

    def forward(self, input):
        identity = input
        # Depthwise convolution
        out = self.depthconv1(input)
        out = self.bn1(out)
        out = self.relu1(out)
        # Pointwise convolution
        out = self.pointwise1(out)
        out = self.bn2(out)
        out = self.relu2(out)
        # Second Depthwise convolution
        out = self.depthconv2(out)
        out = self.bn3(out)
        out = self.relu3(out)
        # Second Pointwise convolution
        out = self.pointwise2(out)
        out = self.bn4(out)
        out = self.relu4(out)
        # Bypass connection
        if self.bypass:
            bypass = self.bypass_conv(identity)
            bypass = self.bypass_bn(bypass)
            bypass = self.relu_bypass(bypass)
            out = self.add(out, bypass)
        return out


class Inverted_Linear_Bottleneck(nn.Module):
    """
    Block involving an inverted connection following a narrow-wide-narrow approach
    in terms of channels. The first step widens the network by an 'expand' factor.
    The following depthwise convolution reduces the number of learnable parameters,
    while the last 1x1 convolution squeezes the network. The last layer does not
    present any ReLU activation (linear output) for channel dimension motivations.
    """
    def __init__(self, in_channels, out_channels, expand, bypass=True):
        super(Inverted_Linear_Bottleneck, self).__init__()
        self.bypass = bypass

        # 1x1 expansion convolution
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=in_channels*expand, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(num_features=in_channels*expand)
        self.relu1 = nn.ReLU6(inplace=False)
        # 3x3 depthwise convolution
        self.depthconv = nn.Conv2d(in_channels=in_channels*expand, out_channels=in_channels*expand, kernel_size=3,
                                   stride=2, padding=1, bias=False, groups=in_channels*expand, padding_mode='zeros')
        self.bn2 = nn.BatchNorm2d(num_features=in_channels*expand)
        self.relu2 = nn.ReLU6(inplace=False)
        # 1x1 linear convolution
        self.conv2 = nn.Conv2d(in_channels=in_channels*expand, out_channels=out_channels, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(num_features=out_channels)
        self.relu3 = nn.ReLU6(inplace=False)
        # Bypass connection
        if self.bypass:
            self.bypass_conv= nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                                kernel_size=1, stride=2, padding=0, groups=1, bias=False)
            self.bypass_bn = nn.BatchNorm2d(num_features=out_channels)
            self.relu_bypass = nn.ReLU6(inplace=False)
            self.add = nemo.quant.pact.PACT_IntegerAdd()


    def forward(self, input):
        identity = input
        # 1x1 expansion convolution
        out = self.conv1(input)
        out = self.bn1(out)
        out = self.relu1(out)
        # 3x3 depthwise convolution
        out = self.depthconv(out)
        out = self.bn2(out)
        out = self.relu2(out)
        # 1x1 linear convolution
        out = self.conv2(out)
        out = self.bn3(out)
        out = self.relu3(out)
        # Bypass connection
        if self.bypass:
            bypass = self.bypass_conv(identity)
            bypass = self.bypass_bn(bypass)
            bypass = self.relu_bypass(bypass)
            out = self.add(out, bypass)
        return out

################################################################################
# PULP-Dronet CNN #
################################################################################
class dronet(nn.Module):
    """
    PULP-DroNet CNN architecture.
    """
    ResBlock = ResBlock
    Depthwise_Separable = Depthwise_Separable
    Inverted_Linear_Bottleneck = Inverted_Linear_Bottleneck
    def __init__(self, depth_mult=1.0, block_class=ResBlock, nemo=False, bypass=True):
        super(dronet, self).__init__()
        self.nemo=nemo # Prepare network for quantization? [True, False]
        first_conv_channels=int(32*depth_mult)
        # First convolution: conv 5x5, 1, 32, 200x200, /2
        self.first_conv = nn.Conv2d(in_channels=1, out_channels=first_conv_channels, kernel_size=5, stride=2, padding=2, dilation=1, groups=1, bias=False, padding_mode='zeros')
        self.bn1 = nn.BatchNorm2d(num_features=first_conv_channels, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu1 = nn.ReLU6(inplace=False)
        # Max Pooling: max_pool 2x2, 32, 32, 100x100, /2
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, return_indices=False, ceil_mode=False)
        # Three blocks, each one doubling the number of channels
        self.Block1 = block_class(first_conv_channels,   first_conv_channels,   bypass=bypass)
        self.Block2 = block_class(first_conv_channels,   first_conv_channels*2, bypass=bypass)
        self.Block3 = block_class(first_conv_channels*2, first_conv_channels*4, bypass=bypass)
        # Dropout layer
        if not self.nemo: self.dropout = nn.Dropout(p=0.5, inplace=False)
        # Fully connected layer
        fc_size = (first_conv_channels*4)*7*7
        self.fc = nn.Linear(in_features=fc_size, out_features=2, bias=False)
        self.sig = nn.Sigmoid()

    def forward(self, input):
        # First convolution
        out = self.first_conv(input)
        out = self.bn1(out)
        out = self.relu1(out)
        # Max pooling
        out = self.pool(out)
        # Three blocks
        out = self.Block1(out)
        out = self.Block2(out)
        out = self.Block3(out)
        # Dropout layer
        if not self.nemo: out = self.dropout(out)
        # Fully connected layer
        out = out.flatten(1)
        out = self.fc(out)
        steer = out[:, 0]
        coll = self.sig(out[:, 1])
        return [steer, coll]
