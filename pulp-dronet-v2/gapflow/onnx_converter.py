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
# File:   config.py                                                             #
# Author: Vlad Niculescu   <vladn@iis.ee.ethz.ch>                               #
#         Lorenzo Lamberti <lorenzo.lamberti@unibo.it>                          #
#         Daniele Palossi  <dpalossi@iis.ee.ethz.ch> <daniele.palossi@idsia.ch> #
# Date:   18.02.2021                                                            #
#-------------------------------------------------------------------------------#

import os
from os.path import join
import sys
import argparse
sys.path.append('../')
import torch
from model.dronet_v2_gapflow import dronet


def create_parser(cfg):
    parser = argparse.ArgumentParser(description='PyTorch PULP-DroNet Testing')
    parser.add_argument('--gapflow_model_weights_original',  
                        help='path to the weights of the network trained on the\
                        original dataset (.pth file)',
                        default=cfg.gapflow_model_weights_original)
    parser.add_argument('--gapflow_model_weights_original_himax', 
                        help='path to the weights of the network trained on the\
                        original+himax dataset (.pth file)',
                        default=cfg.gapflow_model_weights_original_himax)
    parser.add_argument('--gapflow_onnx_export_path',
                        help='folder where onnx models will be exported',  
                        default=cfg.gapflow_onnx_export_path)
    return parser

def main():
    # parse arguments
    global args
    from config import cfg # load configuration with all default values 
    parser = create_parser(cfg)
    args = parser.parse_args()

    # Dronet model trained on the original dataset
    path_to_model1 = join('..',args.gapflow_model_weights_original)
    model1 = dronet()
    state_dict1 = torch.load(path_to_model1, map_location=torch.device('cpu'))
    model1.load_state_dict(state_dict1)
    model1.eval()

    # Dronet model trained on the original+himax dataset
    path_to_model2 = join('..',args.gapflow_model_weights_original_himax)
    model2 = dronet()
    state_dict2 = torch.load(path_to_model2, map_location=torch.device('cpu'))
    model2.load_state_dict(state_dict2)
    model2.eval()

    ############################
    #### Export ONNX Models ####
    ############################
    
    export_path = args.gapflow_onnx_export_path
    if not os.path.exists(export_path):
        os.makedirs(export_path)

    dummy_input = torch.randn(1, 1, 200, 200)
    print('Exporting the ONNX file...')
    torch.onnx.export(model1, dummy_input, join(export_path,'model_original.onnx'))
    torch.onnx.export(model2, dummy_input, join(export_path,'model_original_himax.onnx'))
    print('Export Done!')

if __name__ == '__main__':
    main()
