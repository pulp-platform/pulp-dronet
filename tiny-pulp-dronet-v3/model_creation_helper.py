#-----------------------------------------------------------------------------#
# Copyright(C) 2024 University of Bologna, Italy, ETH Zurich, Switzerland.    #
# All rights reserved.                                                        #
#                                                                             #
# Licensed under the Apache License, Version 2.0 (the "License");             #
# you may not use this file except in compliance with the License.            #
# See LICENSE.apache.md in the top directory for details.                     #
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
# File:    model_creation_helper.py                                           #
# Authors:                                                                    #
#          Michal Barcis    <michal.barcis@tii.ae>                            #
#          Lorenzo Lamberti <lorenzo.lamberti@unibo.it>                       #
#          Daniele Palossi  <dpalossi@iis.ee.ethz.ch>                         #
# Date:    01.03.2024                                                         #
#-----------------------------------------------------------------------------#

import os
import torch
from torchvision import transforms

from config import cfg
from classes import Dataset
from utility import DronetDatasetV3, DronetDatasetV2

import argparse

def create_parser(parser=None):
    if parser is None:
        parser = argparse.ArgumentParser(
            description='PULP-DroNet configuration'
        )
    parser.add_argument('-d', '--data_path', help='path to dataset',
                        default=cfg.data_path)
    parser.add_argument('-m', '--model_weights', default=cfg.model_weights,
                        help='path to the weights of the testing network (.pth file)')
    parser.add_argument('-a', '--arch', metavar='checkpoint.pth', default=cfg.arch,
                        choices=['dronet_dory', 'dronet_autotiler', 'dronet_dory_no_residuals'],
                        help='select the NN architecture backbone:')
    parser.add_argument('--block_type', action="store",
                        choices=["ResBlock", "Depthwise", "Inverted"],
                        default="ResBlock")
    parser.add_argument('--depth_mult', default=cfg.depth_mult, type=float,
                        help='depth multiplier that scales number of channels')
    parser.add_argument('--gpu', help='which gpu to use. Just one at'
                        'the time is supported', default=cfg.gpu)
    parser.add_argument('-j', '--workers', default=cfg.workers, type=int, metavar='N',
                        help='number of data loading workers (default: 4)')
    parser.add_argument('-b', '--batch_size', default=cfg.testing_batch_size, type=int,
                        metavar='N',
                        help='mini-batch size (default: 32), this is the total '
                            'batch size of all GPUs')

    return parser


def select_device(gpu):
    os.environ['CUDA_VISIBLE_DEVICES'] = gpu
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("CUDA/CPU device:", device)
    print("pyTorch version:", torch.__version__)
    return device


def get_dronet(architecture):
    if architecture == 'dronet_dory':
        from model.dronet_v2_dory import dronet
    if architecture == 'dronet_dory_nemo':
        from model.dronet_v2_dory import dronet_nemo as dronet
    elif architecture == 'dronet_dory_no_residuals':
        from model.dronet_v2_dory_no_residuals import dronet
    elif architecture == 'dronet_autotiler':
        from model.dronet_v2_autotiler import dronet
    elif architecture == 'dronet_dory_no_residuals':
        from model.dronet_v2_dory_no_residuals import dronet
    else:
        raise ValueError('Doublecheck the architecture that you are trying to use.\
                            Select one between dronet_dory and dronet_autotiler')
    return dronet

def load_weights_into_network(model_weights_path, net, device):
    if os.path.isfile(model_weights_path):
        if torch.cuda.is_available():
            checkpoint = torch.load(model_weights_path, map_location=device)
            print('loaded checkpoint on cuda')
        else:
            checkpoint = torch.load(model_weights_path, map_location='cpu')
            print('CUDA not available: loaded checkpoint on cpu')
        if 'state_dict' in checkpoint:
            checkpoint = checkpoint['state_dict']
        else:
            print('Failed to find the [''state_dict''] inside the checkpoint. I will try to open it anyways.')
        net.load_state_dict(checkpoint)
    else:
        raise RuntimeError('Failed to open checkpoint. provide a checkpoint.pth.tar file')


def get_dataloader(
    partition,
    dataset_path,
    batch_size,
    num_workers,
):
    dataset = Dataset(dataset_path)
    dataset.initialize_from_filesystem()
    transformations = transforms.Compose([transforms.CenterCrop(200), transforms.ToTensor()])
    # load testing set
    test_dataset = DronetDatasetV3(
        transform=transformations,
        dataset=dataset,
        selected_partition=partition,
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )
    return test_loader


def get_dataloader_v2(
    partition,
    dataset_path,
    batch_size,
    num_workers,
):
    test_dataset = DronetDatasetV2(
        root=os.path.join(dataset_path, partition),
        transform=transforms.ToTensor())
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers
    )
    return test_loader
