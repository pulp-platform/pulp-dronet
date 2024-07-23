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
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
# PULP-dronet
from utility import DronetDatasetV3
from utility import custom_mse, custom_accuracy, custom_bce, custom_loss_v3
from utility import AverageMeter
from utility import regression_as_classification, custom_accuracy_yawrate_thresholded
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
    parser.add_argument('--block_type', action="store", choices=["ResBlock", "Depthwise", "Inverted"], default="ResBlock")
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


def testing_yawrate_thresholded(model, testing_loader, device):
    """ 
    Yaw_rate: trained as regression problem, tested as classification problem (thresholds [-0.1, +0.1])
    Collision: trained/tested as classification problem
    """
    model.eval()
    # # testing metrics
    acc_collision_test = AverageMeter('ACC', ':.3f') 
    acc_yawrate_test = AverageMeter('ACC-yawrate', ':.4f') 

    with tqdm(total=len(testing_loader), desc='Test', disable=not True) as t:
        with torch.no_grad():
            for batch_idx, data in enumerate(testing_loader):
                inputs, labels, filename = data[0].to(device), data[1].to(device), data[2]

                outputs = model(inputs)
                # accuracy for collision
                acc_collision = custom_accuracy(labels, outputs, device)
                # accuracy for regression putting thresholds at [-0.1, +0.1]
                yaw_rate_labels = labels[:,0].squeeze() 
                yaw_rate_pred = outputs[0]
                acc_yawrate = custom_accuracy_yawrate_thresholded(yaw_rate_labels, yaw_rate_pred, device)
                # store values
                acc_collision_test.update(acc_collision.item())
                acc_yawrate_test.update(acc_yawrate.item())

                t.set_postfix({'yr_acc' : acc_yawrate_test.avg, 'col_acc' : acc_collision_test.avg})
                t.update(1)

    print("Testing Acc yaw_rate: %.4f" % acc_yawrate_test.avg, "Acc collision: %.4f" % acc_collision_test.avg)
    return acc_yawrate_test.avg, acc_collision_test.avg

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

def testing_yr_classification(model, testing_loader, device, yawrate_as_classification=True):
    """ 
    Yaw_rate:  classification problem
    Collision: classification problem
    """
    model.eval()
    # # testing metrics
    acc_test = AverageMeter('ACC', ':.3f') 
    bce_test = AverageMeter('BCE', ':.4f') 
    mse_test = AverageMeter('MSE', ':.4f') 
    loss_test = AverageMeter('Loss', ':.4f')
    acc_yawrate_test = AverageMeter('ACC-yawrate', ':.4f') 

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

                if yawrate_as_classification:
                    #yr=yaw-rate
                    yaw_rate_labels = labels[:,0].squeeze() 
                    yaw_rate_pred = outputs[0]
                    acc_yawrate = custom_accuracy_regression_yawrate(yaw_rate_labels, yaw_rate_pred, device)
                    acc_yawrate_test.update(acc_yawrate.item())
                # store values
                acc_test.update(acc.item())
                bce_test.update(bce.item())
                mse_test.update(mse.item())
                loss_test.update(loss.item())
                t.set_postfix({'mse' : mse_test.avg, 'acc' : acc_test.avg})
                t.update(1)

    print("Testing MSE: %.4f" % mse_test.avg, "Acc: %.4f" % acc_test.avg)
    if yawrate_as_classification:
        print("Testing Acc for yaw_rate: %.4f" % acc_yawrate_test.avg, "Acc for collision: %.4f" % acc_test.avg)
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

    # import PULP-DroNet CNN architecture
    if args.arch == 'dronet_dory':
        from model.dronet_v2_dory import dronet, ResBlock, Depthwise_Separable
    elif args.arch == 'dronet_dory_no_residuals':
        from model.dronet_v2_dory_no_residuals import dronet, ResBlock, Depthwise_Separable
    elif args.arch == 'dronet_autotiler':
        from model.dronet_v2_autotiler import dronet
    elif args.arch == 'dronet_dory_no_residuals':
        from model.dronet_v2_dory_no_residuals import dronet
    else: 
        raise ValueError('Doublecheck the architecture that you are trying to use.\
                            Select one between dronet_dory and dronet_autotiler')
                            
    # select the CNN model
    print('You are using a depth multiplier of', args.depth_mult, 'for PULP-Dronet')
    if args.block_type == "ResBlock":
        net = dronet(depth_mult=args.depth_mult, block_class=ResBlock)
    elif args.block_type == "Depthwise":
        net = dronet(depth_mult=args.depth_mult, block_class=Depthwise_Separable)
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


    ## Create dataloaders for PULP-DroNet Dataset
    from dataset_browser.models import Dataset
    dataset = Dataset(args.data_path)
    dataset.initialize_from_filesystem()
    transformations = transforms.Compose([transforms.CenterCrop(200), transforms.ToTensor()])
    # load testing set
    test_dataset = DronetDatasetV3(
        transform=transformations,
        dataset=dataset,
        selected_partition='test',
        remove_yaw_rate_zero=args.remove_yaw_rate_zero,
        labels_preprocessing=False)
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
    testing_yawrate_thresholded(net, test_loader, device)


if __name__ == '__main__':
    main()