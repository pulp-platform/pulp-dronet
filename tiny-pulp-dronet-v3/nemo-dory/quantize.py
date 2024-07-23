
#-----------------------------------------------------------------------------#
# Copyright(C) 2021-2022 ETH Zurich, Switzerland, University of Bologna, Italy#
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
# File:    quantize.py                                                         #
# Author:  Lorenzo Lamberti <lorenzo.lamberti@unibo.it>                       #
#          Daniele Palossi  <dpalossi@iis.ee.ethz.ch>                         #
#          Vlad Niculescu   <vladn@iis.ee.ethz.ch>                            #
# Date:    3.05.2021                                                         #
#-----------------------------------------------------------------------------#

# Description:
# Script of NEMO quantization tool for automatic export of a pytorch-defined NN
# architecture in Dory format
# Input: a pytorch network definition ("--arch=dronet_dory") and a set of
#        pre-trained weights (".pth" file given through '--model_weights').
# Output: - ONNX graph representation file (of the 8-bit quantized network)
#                (Note: this file is comprehensive of NN weights)
#         - Golden activations for all the layers (used by DORY only for checksums)
#
# Brief description of NEMO:
# (more details can be found at: https://github.com/pulp-platform/nemo)
# NEMO operates on three different "levels" of quantization-aware DNN representations,
# all built upon torch.nn.Module and torch.autograd.Function:
# 1. Fake-quantized FQ: replaces regular activations (e.g., ReLU) with
#    quantization-aware ones (PACT) and dynamically quantized weights (with linear
#    PACT-like quantization), maintaining full trainability (similar to the
#    native PyTorch support, but not based on it).
# 2. Quantized-deployable QD: replaces all function with deployment-equivalent
#    versions, trading off trainability for a more accurate representation of
#    numerical behavior on real hardware.
# 3. Integer-deployable ID: replaces all activation and weight tensors used
#    along the network with integer-based ones. It aims at bit-accurate representation
#    of actual hardware behavior. All the quantized representations support mixed-precision
#    weights (signed and asymmetric) and activations (unsigned). The current version of NEMO
#    targets per-layer quantization; work on per-channel quantization is in progress.

#essentials
import sys
import os
from os.path import join
import numpy as np
import argparse
from tqdm import tqdm
#torch
import torch; print('\nPyTorch version in use:', torch.__version__, '\ncuda avail: ', torch.cuda.is_available())
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
#nemo
sys.path.append('/home/lamberti/work/nemo') # if you want to use your custom installation (git clone) instead of pip version
import nemo
from copy import deepcopy
from collections import OrderedDict
from dataset_browser.models import Dataset
# PULP-dronet
from utility import (
    DronetDatasetV3,
    custom_mse,
    custom_accuracy,
    custom_mse_id_nemo,
    get_fc_quantum,
)

def create_parser(cfg):
    parser = argparse.ArgumentParser(description='PyTorch PULP-DroNet quantization with NEMO tool (pulp-platform)')
    parser.add_argument('-d', '--data_path', help='path to dataset',
                        default=cfg.data_path)
    parser.add_argument('--data_path_testing', help='path to dataset')
    parser.add_argument('-m', '--model_weights', default=cfg.model_weights,
                        help='path to the weights of the testing network (.pth file)')
    parser.add_argument('-a', '--arch', metavar='ARCH', default='dronet_dory',
                        choices=['dronet_dory', 'dronet_dory_no_residuals'],
                        help='select the NN architecture backbone. '
                        'Note that dory_autotiler should be quantized with the'
                        'corresponding Greenwaves Technologies tool called NNTools')
    parser.add_argument('--depth_mult', default=cfg.depth_mult, type=float,
                        help='depth multiplier that scales number of channels')                        
    parser.add_argument('--export_path', default=cfg.nemo_export_path,
                        help='folder where the nemo output (onnx and layer activations)'
                        'will be saved')
    parser.add_argument('--onnx_name', default=cfg.nemo_onnx_name,
                        help='the name for the output onnx graph')
    parser.add_argument('--gpu', help='which gpu to use. Just one at'
                        'the time is supported', default=cfg.gpu)
    parser.add_argument('-j', '--workers', default=1, type=int, metavar='N',
                        help='number of data loading workers (default: 4)')
    parser.add_argument('-b', '--batch_size', default=cfg.testing_batch_size, type=int,
                        metavar='N',
                        help='mini-batch size (default: 32), this is the total '
                            'batch size of all GPUs')
    parser.add_argument('--block_type', action="store", choices=["ResBlock", "Depthwise", "Inverted"], default="ResBlock")
    parser.add_argument('--save_quantum', action='store_false', help='append the value of the CNN quantum to the onnx file name')
    parser.add_argument('--test_only_one', action='store_true', help='only one image of the testing set')

    return parser

def clean_directory(export_path):
    '''cleans from NEMO's exported activations'''
    import glob
    types = ('out*', 'in*') # the tuple of file types
    files_grabbed=[]
    for files in types:
        files_grabbed.extend(glob.glob(join(export_path,files)))

    if not files_grabbed:
        print('directory is empty already. Nothing will be removed in:', export_path)
    else:
        print('removing existing activations in folder:', export_path)
        for f in files_grabbed:
            os.remove(f)
            print('removed:', f)

def print_summary(model):
    summary = nemo.utils.get_summary(model, tuple(torch.squeeze(dummy_input_net, 0).size()), verbose=True)
    print(summary['prettyprint'])

def get_intermediate_activations(net, dummy_input_net):
    l = len(list(net.named_modules()))
    buffer_in  = OrderedDict([])
    buffer_out = OrderedDict([])
    hooks = OrderedDict([])
    def get_hk(n):
        def hk(module, input, output):
            buffer_in  [n] = input
            buffer_out [n] = output
        return hk
    for i,(n,l) in enumerate(net.named_modules()):
        hk = get_hk(n)
        hooks[n] = l.register_forward_hook(hk)

    outputs = net(dummy_input_net)
    return buffer_in, buffer_out

def print_summary(model):
    summary = nemo.utils.get_summary(model, (1, 200, 200),verbose=True)
    print(summary['prettyprint'])

def network_size(model):
    summary = nemo.utils.get_summary(model, (1, 200, 200),verbose=True)
    params_size = 0
    for layer_name, layer_info in summary['dict'].items():
        try:
            params_size += abs(layer_info["nb_params"]  * layer_info["W_bits"] / 8. / (1024.))
        except KeyError:
            params_size += abs(layer_info["nb_params"] * 32. / 8. / (1024.))
    return int(params_size)


#pulp dronet
def testing_nemo(model, testing_loader, device, id_stage=False, test_only_one=False):
    model.eval()
    loss_mse, loss_acc = [], []
    test_mse, test_acc = 0.0, 0.0

    with tqdm(total=len(testing_loader), desc='Test', disable=not True) as t:

        with torch.no_grad():
            for batch_idx, data in enumerate(testing_loader):

                if id_stage: #        sample = (_image, _label, _type)
                    data[0] *= 255
                    fc_quantum = get_fc_quantum(args, model)
                    fc_quantum_tensor = fc_quantum.repeat(data[1].size(0))
                # inputs, labels, types = data[0].to(device), data[1].to(device), data[2].to(device)
                inputs, labels, filename = data[0].to(device), data[1].to(device), data[2]
                outputs = model(inputs)

                # change mse only on id_stage
                if not id_stage:
                    mse = custom_mse(labels, outputs, device)
                else:
                    mse = custom_mse_id_nemo(labels, outputs, fc_quantum_tensor, device)

                acc = custom_accuracy(labels, outputs, device)

                # we might have batches without steering or collision samples
                loss_mse.append(mse.item())
                test_mse = sum(loss_mse)/len(loss_mse)
                loss_acc.append(acc.item())
                test_acc = sum(loss_acc)/len(loss_acc)

                t.set_postfix({'mse' : test_mse, 'acc' : test_acc})
                t.update(1)

                if test_only_one: #useful for saving activations from a random image
                    break
    return test_mse, test_acc


#pulp dronet
def test_on_one_image(model, testing_dataset, device, id_stage=False):
    # this function takes just one image and makes a forward pass into the model.
    # it is used for saving the intermediate activations values of the network,
    # and DORY uses them for calculating checksums.

    model.eval()
    with torch.no_grad():
        image = test_dataset[1][0]
        if id_stage:
            image *= 255

        image = torch.reshape(image, (1,1,200,200))
        image = image.to(device)
        outputs = model(image)
    return


def get_quantized_model(model, device, test_loader=None):
    """ test_loader is optional, if not given, no stats will be reported """

    dummy_input_net = torch.randn((1, 1, 200, 200)).to(device) # images are 200x200 px in dronet
    ############################################################################
    # Full Precision
    ############################################################################

    if test_loader is not None:
        test_mse, test_acc = testing_nemo(model, test_loader, device, test_only_one=args.test_only_one)
        model_size = network_size(model)
        print("Full precision MSE: %.4f , Acc: %.4f, Model size: %.2fkB"  % (test_mse,  test_acc, model_size) )

    ############################################################################
    # FakeQuantized (FQ) stage
    ############################################################################

    model_q = nemo.transform.quantize_pact(deepcopy(model), dummy_input=dummy_input_net, remove_dropout=True)
    model_q.change_precision(bits=8, scale_weights=False, scale_activations=True)
    model_q.change_precision(bits=7, scale_weights=True, scale_activations=False)

    if test_loader is not None:
        test_mse, test_acc = testing_nemo(model_q, test_loader, device, id_stage=False, test_only_one=args.test_only_one)
        model_size = network_size(model_q)
        print("FakeQuantized MSE: %.4f , Acc: %.4f, Model size: %.2fkB"  % (test_mse,  test_acc, model_size) )

    ############################################################################
    # QuantizedDeployable (QD) stage
    ############################################################################

    model_q.qd_stage(eps_in=1./255)  # eps_in is the input quantum, and must be set by the user

    if test_loader is not None:
        test_mse, test_acc = testing_nemo(model_q, test_loader, device, id_stage=False, test_only_one=args.test_only_one)
        model_size = network_size(model_q)
        print("QuantizedDeployable MSE: %.4f , Acc: %.4f, Model size: %.2fkB"  % (test_mse,  test_acc, model_size))

    ############################################################################
    # IntegerDeployable (ID) stage
    ############################################################################

    model_q.id_stage()

    if test_loader is not None:
        test_mse, test_acc = testing_nemo(model_q, test_loader, device, id_stage=True, test_only_one = False)
        model_size = network_size(model_q)
        print("IntegerDeployable MSE: %.4f , Acc: %.4f, Model size: %.2fkB"  % (test_mse,  test_acc, model_size))
    return model_q


################################################################################
#### MAIN ####
################################################################################

if __name__ == '__main__':

    # parse arguments
    global args
    from config import cfg # load configuration with all default values
    parser = create_parser(cfg)
    args = parser.parse_args()
    model_weights_path=args.model_weights
    print("Model name:", model_weights_path)

    # select device
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("CUDA/CPU device:", device)
    print("pyTorch version:", torch.__version__)

    # import PULP-DroNet CNN architecture
    if args.arch == 'dronet_dory':
        from model.dronet_v2_dory import dronet, ResBlock, Depthwise_Separable
    elif args.arch == 'dronet_dory_no_residuals':
        from model.dronet_v2_dory_no_residuals import dronet, ResBlock, Depthwise_Separable
    else:
        raise ValueError('Doublecheck the architecture that you are trying to use.\
                          and make sure that you are not tring to use a network that \
                          was intended for GAPflow (NNtool +AutoTiler). You must select \
                          dronet_dory, because dronet_autotiler should be quantized with \
                          the corresponding Greenwaves Technologies tool called NNTools.')

    # select the CNN model
    print('You are using the following building blocks:', args.block_type ,' with a depth multiplier of', args.depth_mult, 'for PULP-Dronet',)
    if args.block_type == "ResBlock":
        model = dronet(depth_mult=args.depth_mult, block_class=ResBlock, nemo=True)
    elif args.block_type == "Depthwise":
        model = dronet(depth_mult=args.depth_mult, block_class=Depthwise_Separable, nemo=True)
        
    # pass to device
    model.to(device)
    dummy_input_net = torch.randn((1, 1, 200, 200)).to(device) # images are 200x200 px in dronet

    # print model structure
    print("model structure summary: \n")
    print_summary(model)

    # if args.init_random_weights:
    #     from utility import init_weights
    #     net.apply(init_weights)

    #load weights
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
            print('Failed to find the [''state_dict''] inside the checkpoint. I will try to open it as a checkpoint.')
        model.load_state_dict(checkpoint)
    else:
        raise RuntimeError('Failed to open checkpoint. provide a checkpoint.pth.tar file')

    ## Create dataloaders for PULP-DroNet Dataset
    transformations = transforms.Compose([transforms.CenterCrop(200), transforms.ToTensor()])
    if not args.data_path_testing:
        args.data_path_testing = args.data_path
    print('Training and Validation set paths:', args.data_path)
    print('Testing set path (you should select the non augmented dataset):', args.data_path_testing)

    dataset_noaug = Dataset(args.data_path_testing)
    dataset_noaug.initialize_from_filesystem()
    # load testing set
    test_dataset = DronetDatasetV3(
        transform=transformations,
        dataset=dataset_noaug,
        selected_partition='test')
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers)


    # quantizing the network

    model_q = get_quantized_model(
        model, device, test_loader
    )

    ############################################################################
    # get activation layers names
    ############################################################################
    # By defining the union of Conv-BN-ReLU, this function extracts the activations
    # (by meaning of output) of each layer so defined.

    names = []
    search_classes=['ReLU6', 'ReLU', 'PACT_IntegerAdd', 'PACT_Act', 'MaxPool2d', 'Linear'] # we need just activations, so Relus, pooling, linears, and adds
    print('The following layers do not belong to search_classes and they will be will be ignored')
    for key, class_name in model.named_modules():
        class_name = str(class_name.__class__).split(".")[-1].split("'")[0]
        if class_name in search_classes:
            names.append(key)
        else:
            print(class_name)

    ############################################################################
    # Export ONNX and Activations
    ############################################################################

    # define onnx and activations save path
    export_path = args.export_path # for both onnx and activations
    export_onnx_path = join(export_path,args.onnx_name)
    # If not existing already, create a new folder for all the NEMO output (ONNX + activations)
    os.makedirs(export_path, exist_ok=True)

    # remove old NEMO's activations
    clean_directory(export_path)

    # export graph
    nemo.utils.export_onnx(export_onnx_path, model_q, model_q, dummy_input_net.shape[1:])
    print('\nExport of ONNX graph was successful\n.')

    # Extract activations buffers
    buf_in, buf_out , _ = nemo.utils.get_intermediate_activations(model_q, test_on_one_image, model_q, test_dataset, device, id_stage = True)

    # Save the input buffer
    t = buf_in['first_conv'][0][-1].cpu().detach().numpy()
    np.savetxt(join(export_path,'input.txt'), t.flatten(), '%.3f', newline=',\\\n', header = 'input (shape %s)' % str(list(t.shape)))

    # Save the output buffers
    for l in range(len(names)):
        t = np.moveaxis(buf_out[names[l]][-1].cpu().detach().numpy(), 0, -1)
        if t.max()>255: print('Warning: activation of layer %d is >255, this will result in incorrect checksums in DORY. This is probably due to an incorrect bitwidth problem (>8bits). NOTE: This is not a problem if the overflow happens in the last layer (Fully connected)!' %(l))
        np.savetxt(join(export_path,'out_layer%d.txt') % l, t.flatten(), '%.3f', newline=',\\\n', header = names[l] + ' (shape %s)' % str(list(t.shape)))

    print('\nExport of golden activations was successful \n')

    network_output_quantum = get_fc_quantum(args, model_q) # This also takes into account ONNX approximation
    print('network_output_quantum (after ONNX rounding):', network_output_quantum)

    if args.save_quantum: 
        with open(join(export_path,'quantum='+str("{:.4f}".format(network_output_quantum.item()))) , 'w') as f:
            f.write('this is the nemo''s quantum')

    print('\nEnd.')
