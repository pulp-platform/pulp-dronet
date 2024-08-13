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
# File:    quantize.py                                                        #
# Authors:                                                                    #
#          Lorenzo Lamberti <lorenzo.lamberti@unibo.it>                       #
#          Francesco Conti <f.conti@unibo.it>                                 #
#          Daniele Palossi  <dpalossi@iis.ee.ethz.ch>                         #
# Date:    01.03.2024                                                         #
#-----------------------------------------------------------------------------#

# Description:
# Script of NEMO quantization tool for automatic export of a pytorch-defined NN
# architecture in Dory format
# Input: a pytorch network definition (--bypass, --depth_mul, --block_type ) and a set of
#        pre-trained weights (".pth" file given through '--model_weights_path').
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
# Add the parent directory to the system path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))) # Now you can import the module from the parent directory
from utility import str2bool # custom function to convert string to boolean in argparse
from tqdm import tqdm
from copy import deepcopy
from collections import OrderedDict
#torch
import torch
from torchvision import transforms
#nemo
sys.path.append('./nemo-dory/nemo/')
try:
    import nemo
    print("nemo imported successfully")
except ModuleNotFoundError:
    print("Failed to import nemo")
    print(sys.path)
# import PULP-DroNet CNN architecture
from model.dronet_v3 import ResBlock, Depthwise_Separable, Inverted_Linear_Bottleneck
from model.dronet_v3 import dronet
from utility import load_weights_into_network
# PULP-dronet dataset
from classes import Dataset
from utility import DronetDatasetV3
# PULP-dronet utilities
from utility import custom_mse, custom_accuracy, custom_mse_id_nemo, get_fc_quantum


def create_parser(cfg):
    parser = argparse.ArgumentParser(description='PyTorch PULP-DroNet quantization with NEMO tool (pulp-platform)')
    # Path to dataset
    # parser.add_argument('-d', '--data_path',
    #                     help='Path to the training dataset',
    #                     default=cfg.data_path,
    #                     metavar='DIRECTORY')
    parser.add_argument('--data_path_testing',
                        help='Path to the testing dataset',
                        metavar='DIRECTORY')
    # Model weights
    parser.add_argument('-w', '--model_weights_path',
                        default=cfg.model_weights_path,
                        help='Path to the weights file for resuming training (.pth file)',
                        metavar='WEIGHTS_FILE')
    # CNN architecture
    parser.add_argument('--bypass',
                        metavar='BYPASS_BRANCH',
                        default=cfg.bypass,
                        type=str2bool,
                        help='Select if you want by-pass branches in the neural network architecture')
    parser.add_argument('--block_type',
                        choices=["ResBlock", "Depthwise", "IRLB"],
                        default="ResBlock",
                        help='Type of blocks used in the network architecture',
                        metavar='BLOCK_TYPE')
    parser.add_argument('--depth_mult',
                        default=cfg.depth_mult,
                        type=float,
                        help='Depth multiplier that scales the number of channels',
                        metavar='FLOAT')
    # Output
    parser.add_argument('--save_quantum',
                        action='store_false',
                        help='append the value of the CNN quantum to the onnx file name')
    parser.add_argument('--export_path',
                        default=cfg.nemo_export_path,
                        help='folder where the nemo output (onnx and layer activations) will be saved')
    parser.add_argument('--onnx_name',
                        default=cfg.nemo_onnx_name,
                        help='the name for the output onnx graph')
    # CPU/GPU params
    parser.add_argument('--gpu',
                        help='Which GPU to use (only one GPU supported)',
                        default=cfg.gpu,
                        metavar='GPU_ID')
    parser.add_argument('-j', '--workers',
                        default=cfg.workers,
                        type=int,
                        metavar='N',
                        help='Number of data loading workers (default: 4)')
    # utilities
    parser.add_argument('-b', '--batch_size', default=cfg.testing_batch_size, type=int,
                        metavar='N',
                        help='mini-batch size (default: 32), this is the total '
                            'batch size of all GPUs')
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

def print_summary(model, dummy_input_net):
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
def test_on_one_image(model, test_dataset, device, id_stage=False):
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

    net_q = nemo.transform.quantize_pact(deepcopy(model), dummy_input=dummy_input_net, remove_dropout=True)
    net_q.change_precision(bits=8, scale_weights=False, scale_activations=True)
    net_q.change_precision(bits=7, scale_weights=True, scale_activations=False)

    if test_loader is not None:
        test_mse, test_acc = testing_nemo(net_q, test_loader, device, id_stage=False, test_only_one=args.test_only_one)
        model_size = network_size(net_q)
        print("FakeQuantized MSE: %.4f , Acc: %.4f, Model size: %.2fkB"  % (test_mse,  test_acc, model_size) )

    ############################################################################
    # QuantizedDeployable (QD) stage
    ############################################################################

    net_q.qd_stage(eps_in=1./255)  # eps_in is the input quantum, and must be set by the user

    if test_loader is not None:
        test_mse, test_acc = testing_nemo(net_q, test_loader, device, id_stage=False, test_only_one=args.test_only_one)
        model_size = network_size(net_q)
        print("QuantizedDeployable MSE: %.4f , Acc: %.4f, Model size: %.2fkB"  % (test_mse,  test_acc, model_size))

    ############################################################################
    # IntegerDeployable (ID) stage
    ############################################################################

    net_q.id_stage()

    if test_loader is not None:
        test_mse, test_acc = testing_nemo(net_q, test_loader, device, id_stage=True, test_only_one = False)
        model_size = network_size(net_q)
        print("IntegerDeployable MSE: %.4f , Acc: %.4f, Model size: %.2fkB"  % (test_mse,  test_acc, model_size))
    return net_q


################################################################################
# MAIN
################################################################################

def main():
    # parse arguments
    global args
    from config import cfg # load configuration with all default values
    parser = create_parser(cfg)
    args = parser.parse_args()

    model_weights_path=args.model_weights_path
    print("Model name:", model_weights_path)

    # select device
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("CUDA/CPU device:", device)
    print("pyTorch version:", torch.__version__)

    # select the CNN model
    print(
        f'You defined PULP-Dronet architecture as follows:\n'
        f'Depth multiplier: {args.depth_mult}\n'
        f'Block type: {args.block_type}\n'
        f'Bypass: {args.bypass}'
    )

    if args.block_type == "ResBlock":
        net = dronet(depth_mult=args.depth_mult, block_class=ResBlock, bypass=args.bypass)
    elif args.block_type == "Depthwise":
        net = dronet(depth_mult=args.depth_mult, block_class=Depthwise_Separable, bypass=args.bypass)
    elif args.block_type == "IRLB":
        net = dronet(depth_mult=args.depth_mult, block_class=Inverted_Linear_Bottleneck, bypass=args.bypass)

    net = load_weights_into_network(args.model_weights_path, net, device)

    # pass to device
    net.to(device)
    dummy_input_net = torch.randn((1, 1, 200, 200)).to(device) # images are 200x200 px in dronet

    # print model structure
    print("model structure summary: \n")
    print_summary(net, dummy_input_net)

    ## Create dataloaders for PULP-DroNet Dataset
    transformations = transforms.Compose([transforms.CenterCrop(200), transforms.ToTensor()])
    # print('Training and Validation set paths:', args.data_path)
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

    net_q = get_quantized_model(
        net, device, test_loader
    )

    ############################################################################
    # Get Activation Layer Names
    ############################################################################

    # Define the classes of layers for which activations should be extracted
    activation_target_classes = ['ReLU6', 'ReLU', 'PACT_IntegerAdd', 'PACT_Act', 'MaxPool2d', 'Linear']

    # Lists to store the names of layers that will be processed or ignored
    layers_to_extract = []
    layers_to_ignore = []

    # Iterate through all the layers in the network and categorize them
    for layer_name, layer_module in net.named_modules():
        layer_class_name = layer_module.__class__.__name__
        if layer_class_name in activation_target_classes:
            layers_to_extract.append(layer_name)
        else:
            layers_to_ignore.append(layer_name)

    # Print the layers that will have their activations extracted
    print(f"Selected layers for activation extraction ({len(layers_to_extract)}):")
    for name in layers_to_extract:
        print(f"  - {name}")

    # Print the layers that will be ignored
    print(f"\nIgnored layers (not part of target classes) ({len(layers_to_ignore)}):")
    for name in layers_to_ignore:
        print(f"  - {name}")


    ############################################################################
    # Export ONNX Model and Extract Activations
    ############################################################################

    # Set up the directory for exporting the ONNX model and activations
    export_path = args.export_path  # Directory to save both the ONNX model and activation files
    onnx_export_path = os.path.join(export_path, args.onnx_name)

    # Ensure the export directory exists; create it if it does not
    os.makedirs(export_path, exist_ok=True)

    # Clean up any existing activation files in the export directory
    clean_directory(export_path)

    # Export the quantized network model to ONNX format
    nemo.utils.export_onnx(onnx_export_path, net_q, net_q, dummy_input_net.shape[1:])
    print('\nONNX model exported successfully.\n')

    # Extract intermediate activations from the network
    activation_buffers_in, activation_buffers_out, _ = nemo.utils.get_intermediate_activations(
        net_q, test_on_one_image, net_q, test_dataset, device, id_stage=True
    )

    # Save the input activations from the first layer to a text file
    first_layer_input_activations = activation_buffers_in['first_conv'][0][-1].cpu().detach().numpy()
    np.savetxt(
        os.path.join(export_path, 'input_activations.txt'),
        first_layer_input_activations.flatten(),
        fmt='%.3f',
        newline=',\\\n',
        header='Input Activations (shape %s)' % str(list(first_layer_input_activations.shape))
    )

    # Save the output activations of each selected layer to separate text files
    for idx, layer_name in enumerate(layers_to_extract):
        layer_output_activations = np.moveaxis(
            activation_buffers_out[layer_name][-1].cpu().detach().numpy(),
            0, -1
        )

        # Check for any overflow in activation values (above 255) and warn if necessary
        if layer_output_activations.max() > 255:
            print(
                f'Warning in layer {idx} ({layer_name}): Activation values exceed 255. '
                'This may indicate a bitwidth issue (>8 bits) and could lead to incorrect checksums in DORY. '
                'Note: Overflow in the last layer (fully connected) is typically not a problem!'
            )

        # Save the activations to a text file
        np.savetxt(
            os.path.join(export_path, f'layer_{idx}_activations.txt'),
            layer_output_activations.flatten(),
            fmt='%.3f',
            newline=',\\\n',
            header=f'{layer_name} Activations (shape {list(layer_output_activations.shape)})'
        )

    print('\nActivation extraction and export completed successfully.\n')

    # Calculate the network output quantum, considering ONNX rounding
    network_output_quantum = get_fc_quantum(args, net_q)
    print('Network output quantum (after ONNX rounding):', network_output_quantum.item())

    # Save the quantum value to a file if specified
    if args.save_quantum:
        quantum_file_path = os.path.join(export_path, f'quantum={network_output_quantum.item():.6f}')
        with open(quantum_file_path, 'w') as quantum_file:
            quantum_file.write("This is the NEMO's quantum")

    ############################################################################
    # Export Network Characteristics to info.txt
    ############################################################################

    info_file_path = os.path.join(export_path, 'info.txt')
    with open(info_file_path, 'w') as info_file:
        info_file.write("Network Information\n")
        info_file.write("===================\n")
        info_file.write(f"Model name: {model_weights_path}\n")
        info_file.write(f"Depth multiplier: {args.depth_mult}\n")
        info_file.write(f"Block type: {args.block_type}\n")
        info_file.write(f"Bypass: {args.bypass}\n")
        info_file.write(f"Network output quantum: {network_output_quantum.item()}\n")
        info_file.write(f"PyTorch version: {torch.__version__}\n")
        info_file.write(f"Export path: {export_path}\n")
        info_file.write(f"ONNX file path: {onnx_export_path}\n")
        info_file.write(f"Testing dataset path: {args.data_path_testing}\n")
    print(f'\nNetwork information saved to {info_file_path}\n')

    print('\nEnd.\n')

if __name__ == '__main__':
    main()