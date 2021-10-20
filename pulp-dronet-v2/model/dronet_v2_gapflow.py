#-------------------------------------------------------------------------------#
# Copyright (C) 2020-2021 ETH Zurich, Switzerland, University of Bologna, Italy.#
# All rights reserved.                                                          #
#                                                                               #
# Licensed under the Apache License, Version 2.0 (the "License");               #
# you may not use this file except in compliance with the License.              #
# See LICENSE.apache.md in the top directory for details.                       #
# You may obtain a copy of the License at                                       #
#                                                                               #
#   http://www.apache.org/licenses/LICENSE-2.0                                  #
#                                                                               #
# Unless required by applicable law or agreed to in writing, software           #
# distributed under the License is distributed on an "AS IS" BASIS,             #
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.      #
# See the License for the specific language governing permissions and           #
# limitations under the License.                                                #
#                                                                               #
# File:   dronet_v2_gapflow.py                                                  #
# Author: Daniele Palossi  <dpalossi@iis.ee.ethz.ch> <daniele.palossi@idsia.ch> #
#         Lorenzo Lamberti <lorenzo.lamberti@unibo.it>                          #
#         Vlad Niculescu   <vladn@iis.ee.ethz.ch>                               #
# Date:   18.02.2021                                                            #
#-------------------------------------------------------------------------------#


import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim


class ResBlock(nn.Module):

    def __init__(self, in_channels, out_channels):
        super(ResBlock, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=2, padding=1, dilation=1, groups=1, bias=True, padding_mode='zeros')
        
        self.conv2 = nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1, dilation=1, groups=1, bias=True, padding_mode='zeros')
        
        self.bypass = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=2, padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros')
        
        self.bn1 = nn.BatchNorm2d(num_features=out_channels, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

        self.bn2 = nn.BatchNorm2d(num_features=out_channels, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

        self.relu1 = nn.ReLU6(inplace=False)

        self.relu2 = nn.ReLU6(inplace=False)

        self.relu3 = nn.ReLU6(inplace=False)


    def forward(self, x):
        identity = x
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        x_bypass = self.bypass(identity)
        x += self.relu3(x_bypass)
        return x


class dronet(nn.Module):
    def __init__(self):
        super(dronet, self).__init__()

        #conv 5x5, 1, 32, 200x200, /2
        self.layer1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=5, stride=2, padding=2, dilation=1, groups=1, bias=True, padding_mode='zeros')

        #max pooling 2x2, 32, 32, 100x100, /2
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, return_indices=False, ceil_mode=False)

        self.resBlock1 = ResBlock(32, 32)

        self.resBlock2 = ResBlock(32, 64)

        self.resBlock3 = ResBlock(64, 128)

        self.dropout = nn.Dropout(p=0.5, inplace=False)

        self.relu = nn.ReLU6(inplace=False) # chek with 3 different

        fc_size = 128*7*7
        self.fc = nn.Linear(in_features=fc_size, out_features=2, bias=True)

        self.sig = nn.Sigmoid()

    def forward(self, x):
        x = self.layer1(x)
        x = self.pool(x)
        x = self.resBlock1(x)
        x = self.resBlock2(x)
        x = self.resBlock3(x)
        x = self.dropout(x)
        x = self.relu(x)
        x = x.flatten(1)
        x = self.fc(x)

        steer = x[:, 0]
        coll = self.sig(x[:, 1])

        return [steer, coll]
