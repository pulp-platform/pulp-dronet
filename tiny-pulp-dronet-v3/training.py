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
# File:    training.py                                                        #
# Authors:                                                                    #
#          Lorenzo Lamberti <lorenzo.lamberti@unibo.it>                       #
#          Daniele Palossi  <dpalossi@iis.ee.ethz.ch>                         #
# Date:    01.03.2024                                                         #
#-----------------------------------------------------------------------------#

# Script Description:
# This script is used to train the weights of the PULP-DroNet CNN.
# You must specify the CNN architecture (--bypass, --depth_mul, --block_type ).
# Additional utils:
#       --early_stopping: When deactivated, this script will save the trained weights
#                     (".pth" files) for all the epochs (i.e., for 100 epochs the
#                     output will be a set of 100 weights).
#                     When activated, the script will save just one set of weights
#                     (the last one, which is not necessarily the best performing one).

# essentials
import os
import sys
import argparse
from utility import str2bool # custom function to convert string to boolean in argparse
import numpy as np
import shutil
from os.path import join
from tqdm import tqdm
import pandas as pd
from datetime import datetime
# torch
import torch
import torchvision.transforms as transforms
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchinfo import summary
# import PULP-DroNet CNN architecture
from model.dronet_v3 import ResBlock, Depthwise_Separable, Inverted_Linear_Bottleneck
from model.dronet_v3 import dronet
from utility import load_weights_into_network
# PULP-dronet dataset
from classes import Dataset
from utility import DronetDatasetV3
# PULP-dronet utilities
from utility import EarlyStopping, init_weights
from utility import custom_mse, custom_accuracy, custom_bce, custom_loss_v3
from utility import AverageMeter
from utility import write_log

def create_parser(cfg):
    """
    Creates and returns an argument parser for the PyTorch PULP-DroNet training.

    Args:
        cfg: Configuration object containing default values for the arguments.

    Returns:
        argparse.ArgumentParser: Configured argument parser with command-line arguments.
    """
    parser = argparse.ArgumentParser(description='PyTorch PULP-DroNet Training')
    # Path to dataset
    parser.add_argument('-d', '--data_path',
                        help='Path to the training dataset',
                        default=cfg.data_path,
                        metavar='DIRECTORY')
    parser.add_argument('--data_path_testing',
                        help='Path to the testing dataset',
                        metavar='DIRECTORY')
    # Select training mode: classification, regression, or both
    parser.add_argument('--partial_training',
                        default=None,
                        choices=[None, 'classification', 'regression'],
                        help=('Leave None to train on both classification and regression. '
                              'Select classification to train only on collision detection. '
                              'Select regression to train only on yaw rate.'),
                        metavar='TRAIN_MODE')
    # Model configuration
    parser.add_argument('-m', '--model_name',
                        default=cfg.model_name,
                        help='Name of the model to be created during training',
                        metavar='STRING')
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
    # Training parameters
    parser.add_argument('--epochs',
                        default=cfg.epochs,
                        type=int,
                        metavar='N',
                        help='Total number of epochs to run')
    parser.add_argument('-b', '--batch_size',
                        default=cfg.training_batch_size,
                        type=int,
                        help=('Mini-batch size (default: 32). This is the total batch size '
                              'across all GPUs.'),
                        metavar='N')
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
    # Training utilities
    parser.add_argument('--resume_training',
                        default=cfg.resume_training,
                        type=str2bool,
                        help='Resume training from a checkpoint',
                        metavar='RESUME')
    parser.add_argument('--hard_mining_train',
                        default=cfg.hard_mining_train,
                        type=str2bool,
                        help='Enable training with hard mining',
                        metavar='HARD_MINING')
    parser.add_argument('--early_stopping',
                        default=cfg.early_stopping,
                        type=str2bool,
                        help='Enable early stopping during training, with patience and delta parameters',
                        metavar='EARLY_STOP')
    parser.add_argument('--patience',
                        default=cfg.patience,
                        type=int,
                        help='Patience for early stopping (in epochs)',
                        metavar='PATIENCE')
    parser.add_argument('--delta',
                        default=cfg.delta,
                        type=float,
                        help='Maximum delta value for early stopping',
                        metavar='DELTA')
    # Learning rate parameters
    parser.add_argument('--lr', '--learning-rate',
                        default=cfg.learning_rate,
                        type=float,
                        metavar='LEARNING_RATE',
                        help='Initial learning rate',
                        dest='lr')
    parser.add_argument('--lr_decay',
                        default=cfg.lr_decay,
                        type=float,
                        help='Learning rate decay (default: 1e-5)',
                        metavar='LR_DECAY')
    # Checkpoint and logging
    parser.add_argument('-c', '--checkpoint_path',
                        default=cfg.checkpoint_path,
                        type=str,
                        metavar='CHECKPOINT_DIR',
                        help='Path to save checkpoint (default: checkpoints)')
    parser.add_argument('--logs_dir',
                        default=cfg.logs_dir,
                        type=str,
                        metavar='LOG_DIR',
                        help='Path to save log files')
    # Verbose output
    parser.add_argument('--verbose',
                        action='store_true',
                        help='Enable verbose output')

    return parser


# Global variables
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
alpha = 1.0
beta = 0.0
early_stop_checkpoint = 'checkpoint.pth'

# change working directory to the folder of the project
working_dir = os.path.dirname(os.path.realpath(__file__))
os.chdir(working_dir)
print('\nworking directory:', working_dir, "\n")

def validate(test_set, net, data_loader, tensorboard_writer, logs_dir, df_valid, epoch, device):
    if test_set=='valid':
        dataset_string = 'Validation'
        prefix = 'valid'
    elif test_set=='testing':
        dataset_string = 'Test'
        prefix = 'test'

    net.eval()
    # validation
    acc_valid = AverageMeter('ACC', ':.3f')
    bce_valid = AverageMeter('BCE', ':.4f')
    mse_valid = AverageMeter('MSE', ':.4f')
    loss_valid = AverageMeter('Loss', ':.4f')
    for obj in [mse_valid, bce_valid, acc_valid, loss_valid]: obj.reset()

    with torch.no_grad():
        for batch_idx, data in enumerate(data_loader):
            inputs, labels = data[0].to(device), data[1].to(device)
            outputs = net(inputs)
            # we might have batches without steering or collision samples
            mse  = custom_mse(labels, outputs, device)
            bce = custom_bce(labels, outputs, device)
            acc = custom_accuracy(labels, outputs, device)
            loss = custom_loss_v3(
                labels, outputs, device,
                partial_training=args.partial_training
            )
            mse_valid.update(mse.item())
            bce_valid.update(bce.item())
            acc_valid.update(acc.item())
            loss_valid.update(loss.item())
    print(dataset_string+' MSE: %.4f' % float(mse_valid.avg*alpha), 'BCE: %.4f' % float(bce_valid.avg*(1.0-beta)), 'Acc: %.4f' % acc_valid.avg)

    # add to tensorboard
    tensorboard_writer.add_scalar(dataset_string+'/Acc', acc_valid.avg, epoch)
    tensorboard_writer.add_scalar(dataset_string+'/BCE', bce_valid.avg, epoch)
    tensorboard_writer.add_scalar(dataset_string+'/MSE', mse_valid.avg, epoch)
    tensorboard_writer.add_scalar(dataset_string+'/LossV3', loss_valid.avg, epoch)

    # append to pandas csv
    to_append=[epoch,  acc_valid.avg, bce_valid.avg, mse_valid.avg, loss_valid.avg]
    series = pd.Series(to_append, index = df_valid.columns)
    df_valid = df_valid.append(series, ignore_index=True)
    df_valid.to_csv(join(logs_dir, prefix+'.csv'), index=False, float_format="%.4f")
    # write string log files
    log_str = dataset_string+' [{0}][{1}/{2}]\t' \
                'Loss {loss.val:.4f} ({loss.avg:.4f})\t' \
                'MSE {mse.val:.3f} ({mse.avg:.3f})\t' \
                'BCE {bce.val:.3f} ({bce.avg:.3f})\t' \
                'ACC {acc.val:.3f} ({acc.avg:.3f})\t'\
        .format(epoch, batch_idx, len(data_loader),
                loss=loss_valid, mse=mse_valid, bce=bce_valid, acc=acc_valid)
    write_log(logs_dir, log_str, prefix=prefix, should_print=False, mode='a', end='\n')
    return df_valid

################################################################################
# MAIN
################################################################################

def main():
    # parse arguments
    global args
    from config import cfg # load configuration with all default values
    parser = create_parser(cfg)
    args = parser.parse_args()

    # parse variables
    save_checkpoints_path = args.checkpoint_path
    model_name = args.model_name
    model_parameters_path = 'model'
    print("Model name:", model_name)

    # select device
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("CUDA/CPU device:", device)
    print("pyTorch version:", torch.__version__)
    # device = 'cpu' # force CPU

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


    if not args.resume_training:
        net.apply(init_weights)
    else: # load previous weights # TO BE TESTED
        net = load_weights_into_network(args.model_weights_path, net, device)

    net.to(device)
    summary(net, input_size=(1, 1, 200, 200))

    # initialize the optimizer for training
    optimizer = optim.Adam(net.parameters(), lr=args.lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=args.lr_decay, amsgrad=False)

    # initialize the early_stopping object
    if args.early_stopping:
        early_stopping = EarlyStopping(patience=args.patience, delta=args.delta, verbose=True, path=save_checkpoints_path+model_name+'/'+early_stop_checkpoint)

    # print model's state_dict and optimizer's state_dict
    if args.verbose:
        print("Model's state_dict:")
        for param_tensor in net.state_dict():
            print(param_tensor, "\t\t\t", net.state_dict()[param_tensor].size())

        print("Optimizer's state_dict:")
        for var_name in optimizer.state_dict():
            print(var_name, "\t", optimizer.state_dict()[var_name])

    # print dataset paths
    if not args.data_path_testing:
        args.data_path_testing = args.data_path
    print('Training set path:', args.data_path)
    print('Testing set path (you should select the non augmented dataset):', args.data_path_testing)

    ##############################################
    # Create dataloaders for PULP-DroNet Dataset #
    ##############################################

    # init training set
    dataset = Dataset(args.data_path)
    dataset.initialize_from_filesystem()
    # init testing set
    dataset_noaug = Dataset(args.data_path_testing)
    dataset_noaug.initialize_from_filesystem()
    # transformations
    transformations = transforms.Compose([transforms.CenterCrop(200), transforms.ToTensor()])

    # load training set
    train_dataset = DronetDatasetV3(
        transform=transformations,
        dataset=dataset,
        selected_partition='train')
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers)
    # load validation set
    validation_dataset = DronetDatasetV3(
        transform=transformations,
        dataset=dataset,
        selected_partition='valid')
    validation_loader = torch.utils.data.DataLoader(
        validation_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers)
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

    # create training directory
    training_dir = join(os.path.dirname(__file__), 'training')
    training_model_dir = join(training_dir, model_name)
    logs_dir = join(training_model_dir, args.logs_dir)
    tensorboard_dir = join(training_model_dir, 'tensorboard_'+ datetime.now().strftime('%b%d_%H:%M:%S'))
    checkpoint_dir = join(training_model_dir, 'checkpoint')

    os.makedirs(logs_dir, exist_ok=True)
    print("Logs directory: ", logs_dir)
    os.makedirs(tensorboard_dir, exist_ok=True)
    print("Tensorboard directory: ", tensorboard_dir)
    os.makedirs(checkpoint_dir, exist_ok=True)
    print("Checkpoints directory: ", checkpoint_dir)

    # write the training/validation/testing paths to logs. By doing so we keep track of which dataset we use (augemented VS not augmented)
    write_log(logs_dir, 'Training data path:\t'   + args.data_path, prefix='train', should_print=False, mode='a', end='\n')
    write_log(logs_dir, 'Validation data path:\t' + args.data_path, prefix='valid', should_print=False, mode='a', end='\n')
    write_log(logs_dir, 'Testing data path:\t'    + args.data_path_testing, prefix='test', should_print=False, mode='a', end='\n')

    #logging utils
    tensorboard_writer = SummaryWriter(log_dir=tensorboard_dir)
    # training
    acc_train = AverageMeter('ACC', ':.3f')
    bce_train = AverageMeter('BCE', ':.4f')
    mse_train = AverageMeter('MSE', ':.4f')
    loss_train = AverageMeter('Loss', ':.4f')
    # dataframes for csv files
    df_train = pd.DataFrame( columns=['Epoch','ACC','BCE','MSE','Loss'])
    df_valid = pd.DataFrame( columns=['Epoch','ACC','BCE','MSE','Loss'])
    df_test= pd.DataFrame( columns=['Epoch','ACC','BCE','MSE','Loss'])

    ############################################################################
    # Train Loop Starts Here
    ############################################################################
    for epoch in range(args.epochs+1):
        for obj in [mse_train, bce_train, acc_train, loss_train]: obj.reset()
        #### TRAINING ####
        print("Epoch: %d/%d" %  (epoch, args.epochs))
        net.train()
        with tqdm(total=len(train_loader), desc='Train', disable=not True) as t:
            for batch_idx, data in enumerate(train_loader):
                inputs, labels = data[0].to(device), data[1].to(device)
                optimizer.zero_grad()
                outputs = net(inputs)
                mse = custom_mse(labels, outputs, device)
                bce = custom_bce(labels, outputs, device)
                acc = custom_accuracy(labels, outputs, device)
                loss = custom_loss_v3(labels, outputs, device, partial_training=args.partial_training)
                loss.backward()
                optimizer.step()
                # store values
                acc_train.update(acc.item())
                bce_train.update(bce.item())
                mse_train.update(mse.item())
                loss_train.update(loss.item())
                t.set_postfix({'loss': loss_train.avg})
                t.update(1)
        # add to tensorboard
        tensorboard_writer.add_scalar('Training/Acc', acc_train.avg, epoch)
        tensorboard_writer.add_scalar('Training/BCE', bce_train.avg, epoch)
        tensorboard_writer.add_scalar('Training/MSE', mse_train.avg, epoch)
        tensorboard_writer.add_scalar('Training/LossV3', loss_train.avg, epoch)

        # append to pandas csv
        to_append=[epoch,  acc_train.avg, bce_train.avg, mse_train.avg, loss_train.avg]
        series = pd.Series(to_append, index = df_train.columns)
        df_train = df_train.append(series, ignore_index=True)
        df_train.to_csv(join(logs_dir, 'train.csv'), index=False, float_format="%.4f")

        # write string log files
        log_str = 'Train [{0}][{1}/{2}]\t' \
                    'Loss {loss.val:.4f} ({loss.avg:.4f})\t' \
                    'MSE {mse.val:.3f} ({mse.avg:.3f})\t' \
                    'BCE {bce.val:.3f} ({bce.avg:.3f})\t' \
                    'ACC {acc.val:.3f} ({acc.avg:.3f})\t'\
            .format(epoch, batch_idx, len(train_loader),
                    loss=loss_train, mse=mse_train, bce=bce_train, acc=acc_train)
        write_log(logs_dir, log_str, prefix='train', should_print=False, mode='a', end='\n')

        #### VALIDATION ####
        df_valid = validate('valid', net, validation_loader, tensorboard_writer, logs_dir, df_valid, epoch, device)
        df_test = validate('testing', net, test_loader, tensorboard_writer, logs_dir, df_test, epoch, device)

        # early_stopping needs the validation loss to check if it has decresed,
        # and if it has, it will make a checkpoint of the current model
        if args.early_stopping:
            val_mse = df_valid['MSE'].iloc[-1]
            val_bce = df_valid['BCE'].iloc[-1]
            early_stopping(val_mse*alpha + val_bce*(1.0-beta), net)
            if early_stopping.early_stop:
                print('Early stopping')
                break
        else: #  save the model each epoch for performing cherry picking (evaluation.py script)
            # torch.save(net.state_dict(), join(checkpoint_dir, model_name+'_'+str(epoch)+'.pth'))
            torch.save(net.state_dict(), join(checkpoint_dir, model_name+'_'+str(epoch)+'.pth'))
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
        torch.save(net.state_dict(), join(training_model_dir,model_name+'_'+str(epoch)+'.pth'))
    print('Parameters saved')

    #### Testing Set ####
    from testing import testing
    test_mse, test_acc = testing(net, test_loader, device)

    # write string log files
    log_str = 'Testing set:\tMSE {mse:.3f}\tACC {acc:.3f}\t'.format(mse=test_mse, acc=test_acc)
    write_log(logs_dir, log_str, prefix='valid', should_print=False, mode='a', end='\n')


if __name__ == '__main__':
    main()