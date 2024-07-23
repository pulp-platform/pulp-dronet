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
# File:    analyzing.py                                                       #
# Author:  Daniele Palossi  <dpalossi@iis.ee.ethz.ch>                         #
#          Lorenzo Lamberti <lorenzo.lamberti@unibo.it>                       #
#          Vlad Niculescu   <vladn@iis.ee.ethz.ch>                            #
# Date:    18.02.2021                                                         #
# TODO: update this comment section at some point
#-----------------------------------------------------------------------------#

# Description:
#

# essentials
from collections import namedtuple
from datetime import datetime
import os
import argparse
import numpy as np
from os.path import join
from tqdm import tqdm
import json
# torch
import torch
import torchfunc
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
# PULP-dronet
from utility import DronetDatasetV3
from utility import custom_mse, custom_accuracy, custom_bce, custom_loss_v3
from utility import AverageMeter

from model_creation_helper import create_parser

import cv2


LayerMetadata = namedtuple('LayerMetadata', 'index, name, layer')


def import_dronet_dory():
    from model.dronet_v2_dory import dronet
    return dronet


def import_dronet_autotiler():
    from model.dronet_v2_autotiler import dronet
    return dronet


def import_dronet_dory_no_residuals():
    from model.dronet_v2_dory_no_residuals import dronet
    return dronet


DRONET_ARCHITECTURE_IMPORTS = {
    'dronet_dory': import_dronet_dory,
    'dronet_autotiler': import_dronet_autotiler,
    'dronet_dory_no_residuals': import_dronet_dory_no_residuals,
}


def main():
    # change working directory to the folder of the project
    working_dir = os.path.dirname(os.path.realpath(__file__))
    os.chdir(working_dir)
    print('\nworking directory:', working_dir, "\n")

    # parse arguments
    parser = create_parser()
    parser.add_argument(
        '--sparsity_out', default=None,
        help="Specify a path where the sparsity results will be saved as json",
    )
    args = parser.parse_args()
    model_weights_path=args.model_weights
    model_name=model_weights_path
    print("Model name:", model_name)

    # select device
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("CUDA/CPU device:", device)
    print("pyTorch version:", torch.__version__)

    # import PULP-DroNet CNN architecture
    dronet = DRONET_ARCHITECTURE_IMPORTS[args.arch]()

    from dataset_browser.models import Dataset
    dataset = Dataset(args.data_path)
    dataset.initialize_from_filesystem()

    ## Create dataloaders for PULP-DroNet Dataset
    transformations = transforms.Compose([transforms.CenterCrop(200), transforms.ToTensor()])
    # load testing set
    test_dataset = DronetDatasetV3(
        transform=transformations,
        dataset=dataset,
        selected_partition='test')
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=args.workers)
    # load the CNN model
    if args.block_type == 'ResBlock':
        block_class = dronet.ResBlock
    elif args.block_type == 'Depthwise':
        block_class = dronet.Depthwise_Separable
    else:
        raise Exception("Unknown block type")

    net = dronet(
        args.depth_mult,
        block_class=block_class,
    )

    net.to(device)

    #load weights into the network
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


    #### ANALYZING ####
    print('model analyzed:', model_weights_path)

    analyze(
        net, checkpoint, test_loader, device, model_name,
        args.batch_size, args.sparsity_out
    )




def analyze(
    net, checkpoint, test_loader, device, model_name, batch_size,
    sparsity_out
):
    tensorboard_log_dir = os.path.join('runs', datetime.now().strftime('analyze_%b%d_%H:%M:%S') + '_'+ model_name)
    tensorboard_writer = SummaryWriter(log_dir=tensorboard_log_dir)

    add_graph(net, test_loader, tensorboard_writer, device)
    # add_average_activation_image(net, test_loader, tensorboard_writer, device)
    sparsity = calculate_sparsity(net, test_loader, tensorboard_writer, device)
    if sparsity_out:
        with open(sparsity_out, 'w') as f:
            json.dump(sparsity, f)

    ### lorenzo's stuff
    # count_weights_zero_layerwise(checkpoint)
    # activations_sparsity_layerwise(net, test_loader, tensorboard_writer, device)
    # activations_sparsity_neuronwise(net, test_loader, tensorboard_writer, device)

def count_weights_zero_layerwise(checkpoint):
    percentages={}
    for key, weights in checkpoint.items():
        if 'weight' in key and ('conv' in key or 'bypass' in key) and ('bn' not in key):
            thresh=1e-20 #arbitrary threshold
            num_non_zero_weights = torch.sum(torch.logical_or(weights>thresh, weights<-thresh))
            tot_num_elements = torch.tensor(torch.numel(weights))
            percentage_non_zero_weights = torch.div(num_non_zero_weights.float(),tot_num_elements.float())
            percentage_zero_weights = 1-percentage_non_zero_weights
            percentages[key]=percentage_zero_weights.item()

    #print all layer percentages
    print('I will print the precentage of weights that are =0. The threshold set is:', thresh)
    for key, percentage in percentages.items():
        print('{:>40}'.format(key), '%.3f' % (percentage*100.), '%')
    return percentages

def get_intermediate_activations(net, test_fn, *test_args, **test_kwargs):
    # from NEMO: https://github.com/pulp-platform/nemo/blob/ed32239efaa256cea061a5426d0c1507c5f005ee/nemo/utils.py#L257
    from collections import OrderedDict
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
    ret = test_fn(*test_args, **test_kwargs)
    for n,l in net.named_modules():
        hooks[n].remove()
    return buffer_in, buffer_out, ret

def activations_sparsity_layerwise(net, test_loader, tensorboard_writer, device):
    # Extract activations buffers
    def testing_single_image(net, inputs):
                outputs = net(inputs)
    def calculate_percentages(buf_out):
        percentages={}
        for key, outputs in buf_out.items():
            if 'relu' in key:
                tot_num_elements = torch.tensor(torch.numel(outputs))
                num_zero_output = torch.sum(outputs==0)
                percentage_zero = torch.div(num_zero_output.float(),tot_num_elements.float())
                # percentage_non_zero = 1-percentage_zero
                percentages[key]=percentage_zero.item()
        return percentages
    samples_num = 0
    print("Running data through the model and saving stats")
    with torch.no_grad():
        for batch_num, data in enumerate(tqdm(test_loader)):
            inputs, labels = data[0].to(device), data[1].to(device)
            samples_num += inputs.size(0)
            buf_in, buf_out , _ = get_intermediate_activations(net, testing_single_image, net, inputs)

            if batch_num==0:
                percentages = calculate_percentages(buf_out)
            else:
                percentages_new = calculate_percentages(buf_out)
                for key, _ in percentages_new.items():
                    percentages[key]+=percentages_new[key]
                    percentages[key]= percentages[key]/2
    #print all layer percentages
    print('I will print the precentage of outputs that are =0 (activation probability).')
    for key, percentage in percentages.items():
        print('{:>40}'.format(key), percentage)
    return percentages

def activations_sparsity_neuronwise(net, test_loader, tensorboard_writer, device):
    # Extract activations buffers
    def testing_single_image(net, inputs):
                outputs = net(inputs)
    def calculate_percentages(buf_out):
        percentages={}
        for key, outputs in buf_out.items():
            if 'relu' in key:
                tot_num_elements = torch.tensor(torch.numel(outputs))
                num_zero_output = torch.sum(outputs==0)
                percentage_zero = torch.div(num_zero_output.float(),tot_num_elements.float())
                # percentage_non_zero = 1-percentage_zero
                percentages[key]=percentage_zero.item()
        return percentages
    samples_num = 0
    print("Running data through the model and saving stats")

    percentages={}
    with torch.no_grad():
        for batch_num, data in enumerate(tqdm(test_loader)):
            inputs, labels = data[0].to(device), data[1].to(device)
            samples_num += inputs.size(0)
            buf_in, buf_out , _ = get_intermediate_activations(net, testing_single_image, net, inputs)

            if batch_num==0: # first iteration
                for key, _ in buf_out.items():
                    if 'relu' in key:
                        activation_map = (buf_out[key]!=0).int()
                        activation_map = torch.sum(activation_map, dim=0) # sum along batch axis
                        percentages[key] = activation_map
            else:
                for key, _ in buf_out.items():
                    if 'relu' in key:
                        activation_map_new = (buf_out[key]!=0).int()
                        activation_map_new = torch.sum(activation_map_new, dim=0) # sum along batch axis
                        percentages[key] = percentages[key] + activation_map_new

    # normalize in [0,1] range
    for key, _ in percentages.items():
        percentages[key] = torch.div(percentages[key],32.*(batch_num+1))

    percentages_layerwise = {}
    for key, _ in percentages.items():
        tot_num_elements = torch.tensor(torch.numel(percentages[key]))
        num_zero_output = torch.sum(percentages[key]==0)
        percentage_zero = torch.div(num_zero_output.float(),tot_num_elements.float())
        # percentage_non_zero = 1-percentage_zero
        percentages_layerwise[key]=percentage_zero.item()

    #print all layer percentages
    print('I will print the precentage of outputs that are =0 for the entire testing set: these neurons are NEVER activated.')
    for key, percentage in percentages_layerwise.items():
        print('{:>40}'.format(key), '%.3f' % (percentage*100.), '%')

    return percentages


def calculate_sparsity(net, test_loader, tensorboard_writer, device):
    """
    As a side effect, it also adds histograms of activations to tensorboard for
    easier analysis.
    """
    threshold = 0

    def summarizer(x, y):
        assert isinstance(y, torch.Tensor)
        if isinstance(x, torch.Tensor):
            x = ("accumulated", (x>threshold).long())
        assert x[0] == "accumulated"
        return ("accumulated", x[1] + (y>threshold).long())

        # return x.sum(0, keepdim=True) + y.sum(0, keepdim=True)
        # return torch.cat((x.flatten().to('cpu'), y.flatten().to('cpu')))

    recorder = torchfunc.hooks.recorders.ForwardOutput(
        reduction=summarizer
    )

    filtered_layers = [
        LayerMetadata(index, layer_name, layer)
        for index, (layer_name, layer) in enumerate(net.named_modules())
        if isinstance(layer, torch.nn.ReLU6)
    ]
    recorder.modules(net, indices=[metadata.index for metadata in filtered_layers])
    samples_num = 0

    print("Running data through the model and saving stats")
    with torch.no_grad():
        for batch_num, data in enumerate(tqdm(test_loader)):
            inputs, labels = data[0].to(device), data[1].to(device)
            samples_num += inputs.size(0)
            assert inputs.size(0) == 1, "The batch size must be 1, otherwise torchfunc is not working properly."
            outputs = net(inputs)

    # format of recorder:
    # recorder[layer_idx_in_recorder][feature_num][width][height]

    print("Processing stats: printing the activation probability")
    # samples_num = 1  # comment this line out to see the average
    result = {}
    for layer_idx_in_recorder, org_data in enumerate(recorder):
        data = org_data[1].true_divide(samples_num)
        layer_metadata = filtered_layers[layer_idx_in_recorder]
        layer_str = f"layer {layer_metadata.name} #{layer_metadata.index}"
        tensorboard_writer.add_histogram(
            f"all_batches_histogram/{layer_str}",
            data,
        )
        # import ipdb; ipdb.set_trace()
        grayscale_image = data
        zeros = (grayscale_image == 0).float()
        tensorboard_writer.add_images(
            f"image(avg)/layer {layer_metadata.name} #{layer_metadata.index}",
            # data[None, :, :, :].true_divide(samples_num),  # adding a dimension for channel a computing average
            torch.stack([zeros, grayscale_image, grayscale_image]),
            dataformats='CNHW',
        )

        # for ident, image in enumerate(org_data[1]):
        #     tensorboard_writer.add_images(
        #         f"image({ident})/layer {layer_metadata.name} #{layer_metadata.index}",
        #         # data[None, :, :, :],  # adding a dimension for channel a computing average
        #         image[None, None, :, :].int().true_divide(image.max()),  # adding a dimension for channel a computing average
        #         dataformats='CNHW',
        #     )

        mean_prob = data.mean()
        tensorboard_writer.add_scalar(
            f"activations_per_layer",
            mean_prob,
            layer_idx_in_recorder,
        )
        count_zeros = (data==0).sum()
        zero_prob = count_zeros.true_divide(data.flatten().size()[0])
        print(f"{layer_str}, non-activation prob: mean {mean_prob} max {data.max()} min {data.min()} zeros {zero_prob}")
        result[layer_str] = zero_prob.item()
    return result


def add_average_activation_image(net, test_loader, tensorboard_writer, device):
    def summarizer(x, y):
        return x+y

    recorder = torchfunc.hooks.recorders.ForwardOutput(reduction=summarizer)

    filtered_layers = [
        LayerMetadata(index, layer_name, layer)
        for index, (layer_name, layer) in enumerate(net.named_modules())
        if isinstance(layer, torch.nn.ReLU6)
    ]
    recorder.modules(net, indices=[metadata.index for metadata in filtered_layers])

    samples_num = 0

    with torch.no_grad():
        for batch_num, data in enumerate(test_loader):
            inputs, labels = data[0].to(device), data[1].to(device)
            samples_num += inputs.size(0)
            outputs = net(inputs)

    # format of recorder:
    # recorder[layer_idx_in_recorder][batch_num][sample_in_batch][feature_num][width][height]


    samples_num = 1  # comment this line out to see the average
    for layer_idx_in_recorder, data in enumerate(recorder):
        layer_metadata = filtered_layers[layer_idx_in_recorder]
        tensorboard_writer.add_images(
            f"image(avg)/layer {layer_metadata.name} #{layer_metadata.index}",
            data[None, :, :, :]/samples_num,  # adding a dimension for channel a computing average
            dataformats='CNHW',
        )


def add_graph(net, test_loader, tensorboard_writer, device):
    it = iter(test_loader)
    data = next(it)
    inputs, labels = data[0].to(device), data[1].to(device)
    tensorboard_writer.add_graph(net, inputs)

if __name__ == '__main__':
    main()
