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
# File:    testing.py                                                         #
# Author:  Daniele Palossi  <dpalossi@iis.ee.ethz.ch>                         #
#          Lorenzo Lamberti <lorenzo.lamberti@unibo.it>                       #
#          Vlad Niculescu   <vladn@iis.ee.ethz.ch>                            #
# Date:    18.02.2021                                                         #
#-----------------------------------------------------------------------------#

# Description:
# This script is used to check the PULP-DroNet CNN performance (Accuracy and RMSE)
# with a set of pre-trained weights.
# You must specify the CNN architecture (dronet_dory or dronet_autotiler) and
# the path to the pre-trained weights that you want to load ('--model_weights').
# The output will be the testing MSE and Accuracy of such network.

# essentials
import os
import sys
import argparse
import numpy as np
from os.path import join
from tqdm import tqdm

# torch
import torch
from torchvision import transforms
from torchinfo import summary

# import PULP-DroNet CNN architecture
from model.dronet_v2_dory import ResBlock, Depthwise_Separable, Inverted_Linear_Bottleneck
from model.dronet_v2_dory import dronet
from utility import load_weights_into_network

# PULP-dronet
from classes import Dataset
from utility import DronetDatasetV3
from utility import custom_mse, custom_accuracy, custom_bce, custom_loss_v3
from utility import AverageMeter
import matplotlib.pyplot as plt
# opencv
import cv2
#nemo
sys.path.append('/home/lamberti/work/nemo') # if you want to use your custom installation (git clone) instead of pip version
import nemo

# change working directory to the folder of the project
working_dir = os.path.dirname(os.path.realpath(__file__))
os.chdir(working_dir)
print('\nworking directory:', working_dir, "\n")

def create_parser(cfg):
    parser = argparse.ArgumentParser(description='PyTorch PULP-DroNet Testing')
    parser.add_argument('-d', '--data_path', help='path to dataset',
                        default=cfg.data_path)
    # parser.add_argument('-s', '--dataset', default=cfg.testing_dataset,
    #                     choices=['original', 'himax'],
                        # help='train on original or original+himax dataset')
    parser.add_argument('-m', '--model_weights', default=cfg.model_weights,
                        help='path to the weights of the testing network (.pth file)')
    parser.add_argument('-a', '--arch', metavar='checkpoint.pth', default=cfg.arch,
                        choices=['dronet_dory', 'dronet_autotiler', 'dronet_dory_no_residuals'],
                        help='select the NN architecture backbone:')
    parser.add_argument('--block_type', action="store", choices=["ResBlock", "Depthwise", "IRLB"], default="ResBlock")
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
    parser.add_argument('--video', action='store_true')
    parser.add_argument('--video_path', default='./output_images/', type=str)
    parser.add_argument('--remove_yaw_rate_zero', action='store_true')

    return parser

def testing(model, testing_loader, device):
    """
    Yaw_rate: trained/tested as regression problem
    Collision: trained/tested as classification problem
    """
    model.eval()
    # # testing metrics
    acc_test = AverageMeter('ACC', ':.3f')
    bce_test = AverageMeter('BCE', ':.4f')
    mse_test = AverageMeter('MSE', ':.4f')
    loss_test = AverageMeter('Loss', ':.4f')

    with tqdm(total=len(testing_loader), desc='Test', disable=not True) as t:
        with torch.no_grad():
            for batch_idx, data in enumerate(testing_loader):
                inputs, labels, filename = data[0].to(device), data[1].to(device), data[2]

                outputs = model(inputs)
                # losses
                mse = custom_mse(labels, outputs, device)
                bce = custom_bce(labels, outputs, device)
                acc = custom_accuracy(labels, outputs, device)
                loss = custom_loss_v3(labels, outputs, device)

                # store values
                acc_test.update(acc.item())
                bce_test.update(bce.item())
                mse_test.update(mse.item())
                loss_test.update(loss.item())
                t.set_postfix({'mse' : mse_test.avg, 'acc' : acc_test.avg})
                t.update(1)

    print("Testing MSE: %.4f" % mse_test.avg, "Acc: %.4f" % acc_test.avg)
    import math
    print("Testing RMSE: %.4f" % math.sqrt(mse_test.avg), "Acc: %.4f" % acc_test.avg)
    return mse_test.avg, acc_test.avg

def list_filenames_labels_outputs(model, testing_loader, device):
    model.eval()

    filenames_list = []
    labels_list = []
    outputs_list = []

    with tqdm(total=len(testing_loader), desc='Test', disable=not True) as t:
        with torch.no_grad():
            for batch_idx, data in enumerate(testing_loader):

                inputs, labels, filename = data[0].to(device), data[1].to(device), data[2]
                outputs = model(inputs)

                filenames_list.extend(list(filename))
                labels_list.extend(labels.cpu().numpy())
                outputs_list.extend(np.vstack((outputs[0].cpu().numpy(), outputs[1].cpu().numpy())).T)
                t.update(1)

    return filenames_list,labels_list, outputs_list

RED = (0, 0, 255)
def export_comparison_groundtruth_output(filenames_list, labels_list, outputs_list, export_images=False, export_video=False,  video_name = 'video', output_directory='./output_images/'):
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    number_of_imgs = len(filenames_list)
    img_array = [] #this variable collects all the images that will create the video
    for image_idx in range(number_of_imgs):
        # --- GET IMAGE and IMAGE PATH---
        img_path = filenames_list[image_idx]

        # --- GET LABELS---
        yaw_rate_label = labels_list[image_idx][0]
        collision_label = labels_list[image_idx][1]

        # --- GET OUTPUTS---
        yaw_rate_output = outputs_list[image_idx][0]
        collision_output = outputs_list[image_idx][1]

        # --- Create cv2 image with overlays ---
        from utility import create_cv2_image
        img = create_cv2_image(img_path,yaw_rate_label, collision_label, yaw_rate_output, collision_output)

        if export_images:
            image_name = join(output_directory, str(image_idx)+".png")
            print('writing image', image_name)
            cv2.imwrite(image_name, img)

        if export_video:
            image_name = os.path.basename(img_path)

            # --- Add image name overlay---
            img = cv2.putText(img,
                            "{}".format(image_name),
                            (150,10) ,
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.3, #font size
                            RED,
                            1)
            img_array.append(img)

    if export_video:
        # --- CREATE VIDEO ---
        # Video name and size
        video_name = join(output_directory, video_name+'.mp4')
        height, width = img.shape[0:2]
        size = (width,height)
        # Define the codec
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        # Create VideoWriter object
        video_out = cv2.VideoWriter(video_name, fourcc, 15, size)
        # Write video
        print('processing the video....')
        for i in range(len(img_array)):
            video_out.write(img_array[i])
        video_out.release()
        print('video saved!')

def plot_outputs(labels_list, outputs_list, depth_mult):
    ground_yaw = [label[0] for label in labels_list]
    pred_yaw = [label[0] for label in outputs_list]
    ground_coll = [label[1] for label in labels_list]
    pred_coll = [label[1] for label in outputs_list]

    plt.figure(figsize=(15, 7))
    plt.plot([ground_coll[i] - pred_coll[i] for i in range(len(ground_coll))], "b")
    plt.title("Ground Truth vs Predicetd Collision")
    plt.xlabel("Step")
    plt.ylabel("Prob. of Collision")
    plt.grid()
    plt.savefig("output_images/pulp_dronet_v3_ResBlock_coll" + str(depth_mult) + ".png")


    plt.figure(figsize=(15, 7))
    plt.stem(ground_yaw, linefmt='b-', markerfmt='bo', basefmt='b-', label="Ground Truth")
    plt.stem(pred_yaw, linefmt='r', markerfmt='ro', basefmt='r-', label="Predicted")
    plt.title("Ground Truth vs Predicetd Yaw Rate")
    plt.xlabel("Step")
    plt.ylabel("Yaw Rate")
    plt.legend()
    plt.grid()
    plt.savefig("output_images/pulp_dronet_v3_ResBlok_yaw" + str(depth_mult) + ".png")


def main():
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



    if args.block_type == "ResBlock":
        net = dronet(depth_mult=args.depth_mult, block_class=ResBlock, bypass=args.bypass)
    elif args.block_type == "Depthwise":
        net = dronet(depth_mult=args.depth_mult, block_class=Depthwise_Separable, bypass=args.bypass)
    elif args.block_type == "IRLB":
        net = dronet(depth_mult=args.depth_mult, block_class=Inverted_Linear_Bottleneck, bypass=args.bypass)

    net = load_weights_into_network(args.model_weights_path, net, args.resume_training, device)

    net.to(device)
    summary(net, input_size=(1, 1, 200, 200))

    ##############################################
    # Create dataloaders for PULP-DroNet Dataset #
    ##############################################

    # init testing set
    dataset_noaug = Dataset(args.data_path_testing)
    dataset_noaug.initialize_from_filesystem()
    # transformations
    transformations = transforms.Compose([transforms.CenterCrop(200), transforms.ToTensor()])

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

    if args.video:
        #### SAVE OUTPUT IMAGES or VIDEO ####
        filenames_list, labels_list, outputs_list = list_filenames_labels_outputs(net, test_loader, device)
        output_directory=args.video_path
        export_comparison_groundtruth_output(filenames_list, labels_list, outputs_list, export_images=False, export_video=True,  video_name = 'video', output_directory=output_directory)
        plot_outputs(labels_list, outputs_list, args.depth_mult)

    #### TESTING ####
    print('model tested:', model_weights_path)
    testing(net, test_loader, device)

if __name__ == '__main__':
    main()