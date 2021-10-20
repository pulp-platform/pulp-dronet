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
# File:   training.py                                                           #
# Author: Daniele Palossi  <dpalossi@iis.ee.ethz.ch> <daniele.palossi@idsia.ch> #
#         Lorenzo Lamberti <lorenzo.lamberti@unibo.it>                          #
#         Vlad Niculescu   <vladn@iis.ee.ethz.ch>                               #
# Date:   18.02.2021                                                            #
#-------------------------------------------------------------------------------#     
#                                    
# Description:
# This script is used to train the weights of the PULP-DroNet CNN.
# You must specify the deployment flow between NEMO-DORY and GAPFlow, and 
# which dataset would you like to use:
#   - original: this dataset is composed of images from Udacity and Zurich Bicycle 
#               datasets. The images have been preprocessed to a 200x200 size 
#               and grayscale format to mimic the HIMAX camera format
#   - original_and_himax: this dataset adds a small set of images acquired with 
#                         the HIMAX camera (on-board of the nano drone). 
#                         It is used to help the network generalizing better.
# Extra:
# '--early_stopping': When deactivated, this script will save the trained weights 
#                     (".pth" files) for all the epochs (i.e., for 100 epochs the 
#                     output will be a set of 100 weights).
#                     When activated, the script will save just one set of weights
#                     (the last one, which is not necessarily the best performing one).  

# essentials
import os
import argparse
import numpy as np
import shutil
from os.path import join
from tqdm import tqdm
import pandas as pd
# torch
import torch
import torchvision.transforms as transforms
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchsummary import summary
# PULP-dronet
from utility import EarlyStopping, init_weights
from utility import DronetDataset
from utility import custom_mse, custom_accuracy, custom_bce, custom_loss
from utility import AverageMeter, write_log

def create_parser(cfg):
    parser = argparse.ArgumentParser(description='PyTorch PULP-DroNet Training')
    parser.add_argument('-d', '--data_path', help='path to dataset',
                        default=cfg.data_path)
    parser.add_argument('-s', '--dataset', default=cfg.training_dataset,
                        choices=['original', 'original_and_himax'], 
                        help='train on original or original+himax dataset')
    parser.add_argument('-m', '--model_name', default=cfg.model_name, 
                        help='model name that is created when training')
    parser.add_argument('-w', '--model_weights', default=cfg.model_weights, 
                        help='path to the weights for resuming training(.pth file)')
    parser.add_argument('-f', '--flow', metavar='DEPLOYMENT_FLOW', default=cfg.flow,
                        choices=['nemo_dory', 'gapflow'],
                        help='select the desired deployment flow between        \
                            nemo_dory and the GreenWaves GAPflow. Side effect:  \
                            this will change the network topology')
    parser.add_argument('--gpu', help='which gpu to use. Just one at'
                        'the time is supported', default=cfg.gpu)
    parser.add_argument('-j', '--workers', default=cfg.workers, type=int, metavar='N',
                        help='number of data loading workers (default: 4)')
    parser.add_argument('--epochs', default=cfg.epochs, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('-b', '--batch_size', default=cfg.training_batch_size, type=int,
                        metavar='N',
                        help='mini-batch size (default: 32), this is the total '
                            'batch size of all GPUs')
    parser.add_argument('--lr', '--learning-rate', default=cfg.learning_rate, type=float,
                        metavar='LR', help='initial learning rate', dest='lr')
    parser.add_argument('--lr_decay', default=cfg.lr_decay, type=float,
                        help='learning rate decay (default: 1e-5)')
    parser.add_argument('-c', '--checkpoint_path', default=cfg.checkpoint_path, type=str, metavar='PATH',
                        help='path to save checkpoint (default: checkpoints)')
    parser.add_argument('--hard_mining_train', default=cfg.hard_mining_train, type=bool,
                        help='do training with hard mining')
    parser.add_argument('--early_stopping', default=cfg.early_stopping, type=bool,
                        help='early stopping at training time, with (patience,delta) parameters')
    parser.add_argument('--patience', default=cfg.patience, type=int,
                        help='patience of early stopping at training time, value in epochs')
    parser.add_argument('--delta', default=cfg.delta, type=float,
                        help='max delta value for early stopping at training time')
    parser.add_argument('--resume_training', default=cfg.resume_training, type=bool, metavar='PATH',
                        help='want to resume training?')
    parser.add_argument('--verbose', action='store_true', help='verbose prints on')
    parser.add_argument('--logs_path', default=cfg.logs_path, type=str, metavar='PATH',
                        help='path to save log files')              
    return parser

# Global variables
device = torch.device("cpu")
alpha = 1.0
beta = 0.0
early_stop_checkpoint = 'checkpoint.pth'

# change working directory to the folder of the project
working_dir = os.path.dirname(os.path.realpath(__file__))
os.chdir(working_dir)
print('\nworking directory:', working_dir, "\n")

################################################################################
# MAIN
################################################################################

def main():
    # parse arguments
    global args
    from config import cfg # load configuration with all default values 
    parser = create_parser(cfg)
    args = parser.parse_args()
    save_checkpoints_path = args.checkpoint_path
    model_name=args.model_name
    model_parameters_path = 'model'
    print("Model name:", model_name)

    # select device
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("CUDA/CPU device:", device)
    print("pyTorch version:", torch.__version__)

    # import PULP-DroNet CNN architecture
    if args.flow == 'nemo_dory':
        from model.dronet_v2_nemo_dory import dronet
    elif args.flow == 'gapflow':
        from model.dronet_v2_gapflow import dronet
    else: 
        raise ValueError('Doublecheck the deployment flow that you are trying to use.\
                            Select one between nemo_dory and gapflow')
    print('You are using the', args.flow ,'deployment flow\n')

    # define training and validation dataset paths
    if args.dataset == 'original':
        training_data_path   = join(args.data_path, "training/")
        validation_data_path = join(args.data_path, "validation/")
    elif args.dataset == 'original_and_himax':
        training_data_path   = join(args.data_path, "fine_tuning/training/")
        validation_data_path = join(args.data_path, "validation/")
    else:
        raise ValueError('they only choices for dataset are: original or original_and_himax')
    print('You are training on the', args.dataset ,'dataset\n')

    ## Create dataloaders for PULP-DroNet Dataset
    # load training set
    train_dataset = DronetDataset(
        root=training_data_path,
        transform=transforms.ToTensor())
    train_loader = torch.utils.data.DataLoader(
        train_dataset, 
        batch_size=args.batch_size, 
        shuffle=True, 
        num_workers=args.workers)

    # load validation set
    validation_dataset = DronetDataset(
        root=validation_data_path,
        transform=transforms.ToTensor())
    validation_loader = torch.utils.data.DataLoader(
        validation_dataset, 
        batch_size=args.batch_size, 
        shuffle=False, 
        num_workers=args.workers)
    
    # select the CNN model
    net = dronet()
    # initialize weights and biases for training
    if not args.resume_training:
        net.apply(init_weights) 
    else: # load previous weights # TO BE TESTED
        if os.path.isfile(args.model_weights):
            if torch.cuda.is_available():
                checkpoint = torch.load(args.model_weights, map_location=device)
                print('loaded checkpoint on cuda')
            else:
                checkpoint = torch.load(args.model_weights, map_location='cpu')
                print('CUDA not available: loaded checkpoint on cpu')
            if 'state_dict' in checkpoint:
                checkpoint = checkpoint['state_dict']
            else:
                print('Failed to find the [''state_dict''] inside the checkpoint. I will try to open it anyways.')
            net.load_state_dict(checkpoint)
        else: 
            raise RuntimeError('Failed to open checkpoint. provide a checkpoint.pth.tar file')
        
    net.to(device)

    # initialize the optimizer for training
    optimizer = optim.Adam(net.parameters(), lr=args.lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=args.lr_decay, amsgrad=False)

    # initialize the early_stopping object
    if args.early_stopping:
        early_stopping = EarlyStopping(patience=args.patience, delta=args.delta, verbose=True, path=save_checkpoints_path+model_name+'/'+early_stop_checkpoint)

    # create temporary folder for checkpoints
    if not os.path.exists(save_checkpoints_path):
        os.mkdir(save_checkpoints_path)
    if not os.path.exists(save_checkpoints_path+model_name):
        os.mkdir(join(save_checkpoints_path,model_name))
        print("Directory", join(save_checkpoints_path,model_name), "created")
    else:    
        print("Directory", join(save_checkpoints_path,model_name), "already exists")

    if args.verbose: 
        # Print model's state_dict
        print("Model's state_dict:")
        for param_tensor in net.state_dict():
            print(param_tensor, "\t\t\t", net.state_dict()[param_tensor].size())
        # Print optimizer's state_dict
        print("Optimizer's state_dict:")
        for var_name in optimizer.state_dict():
            print(var_name, "\t", optimizer.state_dict()[var_name])
        #Print summary
        print('Model summary:')
        summary(net, (1, 200, 200))

    #logging utils
    tensorboard_writer = SummaryWriter()
    os.makedirs(args.logs_path, exist_ok=True)
    # training
    acc_train = AverageMeter('ACC', ':.3f') 
    bce_train = AverageMeter('BCE', ':.4f') 
    mse_train = AverageMeter('MSE', ':.4f') 
    loss_train = AverageMeter('Loss', ':.4f')
    # validation
    acc_valid = AverageMeter('ACC', ':.3f')     
    bce_valid = AverageMeter('BCE', ':.4f') 
    mse_valid = AverageMeter('MSE', ':.4f') 
    loss_valid = AverageMeter('Loss', ':.4f')
    # dataframes for csv files
    df_train = pd.DataFrame( columns=['Epoch','ACC','BCE','MSE','Loss'])
    df_valid = pd.DataFrame( columns=['Epoch','ACC','BCE','MSE','Loss'])

    ############################################################################
    # Train Loop Starts Here
    ############################################################################
    for epoch in range(args.epochs+1):
        for obj in [mse_train, bce_train, acc_train, loss_train]: obj.reset()
        for obj in [mse_valid, bce_valid, acc_valid, loss_valid]: obj.reset()
        #### TRAINING ####
        print("Epoch: %d/%d" %  (epoch, args.epochs))
        net.train()
        running_loss, average_loss = [], 0.0
        with tqdm(total=len(train_loader), desc='Train', disable=not True) as t:
            for batch_idx, data in enumerate(train_loader):
                inputs, labels, types = data[0].to(device), data[1].to(device), data[2].to(device)
                optimizer.zero_grad()
                outputs = net(inputs)
                acc, acc_not_invalid = custom_accuracy(labels, outputs, types, device) # not needed for training, we just log it            
                bce, bce_not_invalid = custom_bce(labels, outputs, types, device)      # not needed for training, we just log it
                mse, mse_not_invalid = custom_mse(labels, outputs, types, device)      # not needed for training, we just log it
                loss = custom_loss(labels, outputs, types, epoch, args, device)
                loss.backward()
                optimizer.step()
                # store values
                if acc_not_invalid == 1: acc_train.update(acc.item())
                if bce_not_invalid == 1: bce_train.update(bce.item())                    
                if mse_not_invalid == 1: mse_train.update(mse.item())
                loss_train.update(loss.item())
                t.set_postfix({'loss': loss_train.avg})
                t.update(1)
        # add to tensorboard
        tensorboard_writer.add_scalar('Training/Acc', acc_train.avg, epoch)
        tensorboard_writer.add_scalar('Training/BCE', bce_train.avg, epoch)
        tensorboard_writer.add_scalar('Training/MSE', mse_train.avg, epoch)
        tensorboard_writer.add_scalar('Training/LossV2', loss_train.avg, epoch)

        # append to pandas csv
        to_append=[epoch,  acc_train.avg, bce_train.avg, mse_train.avg, loss_train.avg]
        series = pd.Series(to_append, index = df_train.columns)
        df_train = df_train.append(series, ignore_index=True)
        df_train.to_csv(join(args.logs_path, 'train.csv'), index=False, float_format="%.4f")

        #### VALIDATION ####
        net.eval()
        with torch.no_grad():
            for batch_idx, data in enumerate(validation_loader):
                inputs, labels, types = data[0].to(device), data[1].to(device), data[2].to(device)
                outputs = net(inputs)
                # we might have batches without steering or collision samples
                acc, acc_not_invalid = custom_accuracy(labels, outputs, types, device)
                bce, bce_not_invalid = custom_bce(labels, outputs, types, device)
                mse, mse_not_invalid = custom_mse(labels, outputs, types, device)
                loss = custom_loss(labels, outputs, types, epoch, args, device)
                if acc_not_invalid == 1: acc_valid.update(acc.item())
                if bce_not_invalid == 1: bce_valid.update(bce.item())                    
                if mse_not_invalid == 1: mse_valid.update(mse.item())
                loss_valid.update(loss.item())
        print('Validation MSE: %.4f' % float(mse_valid.avg*alpha), 'BCE: %.4f' % float(bce_valid.avg*(1.0-beta)), 'Acc: %.4f' % acc_valid.avg)

        # add to tensorboard
        tensorboard_writer.add_scalar('Validation/Acc', acc_valid.avg, epoch)
        tensorboard_writer.add_scalar('Validation/BCE', bce_valid.avg, epoch)
        tensorboard_writer.add_scalar('Validation/MSE', mse_valid.avg, epoch)
        tensorboard_writer.add_scalar('Validation/LossV3', loss_valid.avg, epoch)
        # append to pandas csv
        to_append=[epoch,  acc_valid.avg, bce_valid.avg, mse_valid.avg, loss_valid.avg]
        series = pd.Series(to_append, index = df_valid.columns)
        df_valid = df_valid.append(series, ignore_index=True)
        df_valid.to_csv(join(args.logs_path, 'valid.csv'), index=False, float_format="%.4f")

        # early_stopping needs the validation loss to check if it has decresed, 
        # and if it has, it will make a checkpoint of the current model
        if args.early_stopping:
            early_stopping(mse_valid.avg*alpha + bce_valid.avg*(1.0-beta), net)
            if early_stopping.early_stop:
                print('Early stopping')
                break
        else: #  save the model each epoch for performing cherry picking (evaluation.py script)
            torch.save(net.state_dict(), join(join(save_checkpoints_path,model_name), model_name+'_'+str(epoch)+'.pth'))
        print('Parameters saved')
    print('Training Finished')

    # Save final model, if we use early-stopping we copy the last checkpoint
    if args.early_stopping:
        shutil.copyfile(join(save_checkpoints_path,model_name, early_stop_checkpoint) , join(model_parameters_path,model_name+'_'+str(epoch)+'.pth'))
        # removing temporary checkpoints
        if os.path.exists(save_checkpoints_path+model_name):
            shutil.rmtree(save_checkpoints_path+model_name)
            print('Checkpoint folder', join(save_checkpoints_path,model_name), 'removed')
    else:
        torch.save(net.state_dict(), join(model_parameters_path,model_name+'_'+str(epoch)+'.pth'))
    print('Parameters saved')

if __name__ == '__main__':
    main()
