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
# File:    utility.py                                                         #
# Authors:                                                                    #
#          Lorenzo Lamberti <lorenzo.lamberti@unibo.it>                       #
#          Daniele Palossi  <dpalossi@iis.ee.ethz.ch>                         #
# Date:    01.03.2024                                                         #
#-----------------------------------------------------------------------------#

# essentials
import numpy as np
import pandas as pd
from os.path import join
from PIL import Image
# torch
import torch
import torch.nn as nn
import torch.nn.functional as F

import os
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

################################################################################
# Dataset Loader
################################################################################
class DronetDatasetV3(Dataset):
    """Dronet dataset."""

    def __init__(self, transform=None, dataset=None, selected_partition=None, remove_yaw_rate_zero=False):
        """
        Args:
            root (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.transform = transform
        self.filenames = []
        self.labels = [[],[]]
        self.dataset= dataset
        self.selected_partition = selected_partition

        assert selected_partition in ['train', 'valid', 'test'], 'Error in the partition name you selected'

        for acquisition in self.dataset.acquisitions:
            names, partition, yaw_rate, collision = self.read_csv_label(acquisition)

            # filter images by partition
            data_filter = (partition == selected_partition)
            names = names[data_filter]
            partition = partition[data_filter]
            yaw_rate = yaw_rate[data_filter]
            collision = collision[data_filter]
            # normalize yaw rate
            yaw_rate = self.normalize_yaw_rate(yaw_rate, 90)

            if remove_yaw_rate_zero:
                # print('before',yaw_rate.size)
                zeros = (yaw_rate == np.zeros(yaw_rate.size))
                yaw_rate = np.delete(yaw_rate, zeros)
                names = np.delete(names, zeros)
                collision = np.delete(collision, zeros)
                # print('after',yaw_rate.size, 'deleted', zeros.sum())

            # append labels and filenames to a list
            self.filenames.extend(names)
            self.labels[0].extend(yaw_rate)
            self.labels[1].extend(collision)

            # check if number of images is matching number of labels:
            if not (len(self.filenames) == len(self.labels[0]) == len(self.labels[1])):
                raise RuntimeError("Mismatch in the number of images and labels: images %d, labels_yaw %d, labels_collision %d" % (len(self.filenames), len(self.labels[0]), len(self.labels[1])))

    def __len__(self):
        # check if number of images is matching number of labels:
        if len(self.filenames) == len(self.labels[0]) == len(self.labels[1]):
            return len(self.filenames)
        else:
            raise RuntimeError("DronetDataset size error: filenames %d, label_yaw %d, label_collision %d", len(self.filenames), len(self.labels[0]), len(self.labels[1]))

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        with open(self.filenames[idx], 'rb') as f:
            _image = Image.open(f)
            _image.load()
        _label = [self.labels[0][idx], self.labels[1][idx]]  #[yaw_rate, collision=0/1]
        _filename = self.filenames[idx]

        if self.transform:
            _image = self.transform(_image)
            _label = torch.tensor(_label, dtype=torch.float32)

        sample = (_image, _label, _filename)
        return sample

    def normalize_yaw_rate(self, yaw_rate, normalization=90):
        return yaw_rate/normalization

    def read_csv_label(self, acquisition):
        filename = join(acquisition.path, acquisition.PARTITIONED_LABELS_FILENAME)
        df = pd.read_csv(filename)

        try:
            filenames = (acquisition.images_dir_path + os.sep + df['filename']).to_numpy(dtype=str)
            partition = df['partition'].to_numpy(dtype=str)
            yaw_rate = df['label_yaw_rate'].to_numpy(dtype=float)
            collision = df['label_collision'].to_numpy(dtype=int)
            return filenames, partition, yaw_rate, collision

        except OSError as e:
            print("No labels found in dir", filename)

class DronetDatasetV2(Dataset):
    """Dronet dataset."""

    def __init__(self, root, transform=None):
        """
        Args:
            root (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.root = root
        self.transform = transform
        self.filenames = []
        self.labels = []
        self.types = []

        dirs = os.listdir(root)
        # sorted directory names
        for dir in sorted(dirs):
            full_dir_path = os.path.join(root, dir)
            for sub_root, sub_dirs, files in os.walk(full_dir_path):
                # sorted file names
                for file in sorted(files):
                    # print(os.path.join(full_dir_path, file))
                    if file.endswith(".jpg"):
                        self.filenames.append(os.path.join(full_dir_path+"/images", file))
                    if file.endswith(".csv"):
                        l_steering = self.read_csv_label(os.path.join(full_dir_path, file))
                        self.labels.extend(l_steering)
                        self.types.extend([0]*len(l_steering))
                    if file.endswith(".txt"):
                        l_collision = self.read_txt_label(os.path.join(full_dir_path, file))
                        self.labels.extend(l_collision)
                        self.types.extend([1]*len(l_collision))
                    if not (file.endswith(".jpg") or file.endswith(".csv") or file.endswith(".txt")):
                        print('WARNING: file not loaded', os.path.join(full_dir_path, file))

    def __len__(self):
        if len(self.filenames) == len(self.labels) == len(self.types):
            return len(self.filenames)
        else:
            print("DronetDataset size error", len(self.filenames))


    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        with open(self.filenames[idx], 'rb') as f:
            _image = Image.open(f)
            _image.load()
        _label = self.labels[idx]
        _type = self.types[idx]
        _filename = self.filenames[idx]

        if self.transform:
            _image = self.transform(_image)
            _label = torch.tensor(_label, dtype=torch.float32)

        sample = (_image, _label, _type, _filename)

        return sample

    # collision
    def read_txt_label(self, filename):
        try:
            return np.loadtxt(filename, usecols=0)
        except OSError as e:
            print("No labels found in dir", filename)

    # steering
    def read_csv_label(self, filename):
        try:
            return np.loadtxt(filename, usecols=6, delimiter=',', skiprows=0)
        except OSError as e:
            print("No labels found in dir", filename)


################################################################################
# Logging
################################################################################
class AverageMeter(object):
    """
    Computes and stores the average and current value
    Copied from: https://github.com/pytorch/examples/blob/master/imagenet/main.py
    """
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt # format. example ':6.2f'
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)

def write_log(logs_path, log_str, prefix='train', should_print=True, mode='a', end='\n'):
    # Arguments:
    #   - prefix: the name of the log file:  logs_path/'prefix'.log
    with open(join(logs_path, '%s.log' % prefix), mode) as fout:
        fout.write(log_str + end)
        fout.flush()
    if should_print:
        print(log_str)

################################################################################
# Network Initialization
################################################################################

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
    return net

def rename_and_save_checkpoint(weights_path):
    """
    Rename layers in the state dictionary of a checkpoint and save it to a new file.
    If you change the definition of the block names in dronet_v3.py, you need to rename the layers in the pre trained
    checkpoint files as well!

    Parameters:
        weights_path (str): Path to the original .pth file.

    Returns:
        str: Path to the new weights file.

    Example usage:
        weights_path = "./model/tiny-pulp-dronet-v3-dw-pw-0.125.pth"
        new_weights_path = rename_and_save_checkpoint(weights_path)
    """

    # Define old and new layer names
    old_layer_names = ['resBlock1', 'resBlock2', 'resBlock3']
    new_layer_names = ['Block1', 'Block2', 'Block3']

    # Split the path into directory and filename
    directory, filename = os.path.split(weights_path)
    # Create a new filename by prefixing "new_"
    new_filename = "new_" + filename
    # Join the directory with the new filename to get the new path
    new_weights_path = os.path.join(directory, new_filename)

    # Load the checkpoint
    checkpoint = torch.load(weights_path)

    # Extract the state dictionary
    state_dict = checkpoint['state_dict'] if 'state_dict' in checkpoint else checkpoint

    # Rename layers or parameters
    new_state_dict = {}
    for key, value in state_dict.items():
        for old_layer_name, new_layer_name in zip(old_layer_names, new_layer_names):
            new_key = key.replace(old_layer_name, new_layer_name)
            new_state_dict[new_key] = value

    # Remove all entries with keys that contain any of the old layer names
    keys_to_remove = [key for key in new_state_dict.keys() if any(old_layer_name in key for old_layer_name in old_layer_names)]
    for key in keys_to_remove:
        del new_state_dict[key]

    # Update the checkpoint with the new state dictionary
    if 'state_dict' in checkpoint:
        checkpoint['state_dict'] = new_state_dict
    else:
        checkpoint = new_state_dict

    # Save the modified checkpoint
    torch.save(checkpoint, new_weights_path)
    print(f"Renamed layers and saved the new model to {new_weights_path}")

    return new_weights_path



################################################################################
# Loss Functions and Evaluation Metrics
################################################################################

# steering type=0
def custom_mse(y_true, y_pred, device): # Mean Squared Error
    target_s = y_true[:,0].squeeze()
    input_s  = y_pred[0]
    loss_MSE = nn.MSELoss(size_average=None, reduce=None, reduction='mean')
    output_MSE = loss_MSE(input_s, target_s).to(device)
    return output_MSE


# collision type=1
def custom_accuracy(y_true, y_pred, device):

    tp = 0.0
    tn = 0.0
    fp = 0.0
    fn = 0.0
    acc = torch.zeros(1, 1, dtype=torch.float32, requires_grad=False)
    pre = 0.0
    rec = 0.0
    f1 = 0.0

    pred =  y_pred[1] >= 0.5
    truth = y_true[:,1].squeeze() >= 0.5
    tp += pred.mul(truth).sum(0).float()
    tn += (~pred).mul(~truth).sum(0).float()
    fp += pred.mul(~truth).sum(0).float()
    fn += (~pred).mul(truth).sum(0).float()
    acc = (tp + tn).sum() / (tp + tn + fp + fn).sum()
    pre = tp / (tp + fp)
    rec = tp / (tp + fn)
    f1 = (2.0 * tp) / (2.0 * tp + fp + fn)
    # avg_pre = nanmean(pre)
    # avg_rec = nanmean(rec)
    # avg_f1 = nanmean(f1)
    return acc

# collision type=1
def custom_bce(y_true, y_pred, device): # Debunking loss functions

    output_BCE = torch.zeros(1, 1, dtype=torch.float32, requires_grad=False)

    target_c = y_true[:,1].squeeze()
    input_c  = y_pred[1]
    loss_BCE = nn.BCELoss(weight=None, size_average=None, reduce=None, reduction='mean')
    output_BCE = loss_BCE(input_c, target_c).to(device)
    return output_BCE


# Labels: steering = 0, collision = 1
def custom_loss_v3(y_true, y_pred, device, partial_training=None):

    output_MSE = torch.zeros(1, 1, dtype=torch.float32, requires_grad=True).to(device)
    output_BCE = torch.zeros(1, 1, dtype=torch.float32, requires_grad=True).to(device)

    # yaw_rate
    target_s = y_true[:,0].squeeze()
    input_s  = y_pred[0]
    loss_MSE = nn.MSELoss(size_average=None, reduce=None, reduction='mean')
    output_MSE = loss_MSE(input_s, target_s).to(device)
    # collision
    target_c = y_true[:,1].squeeze()
    input_c  = y_pred[1]
    loss_BCE = nn.BCELoss(weight=None, size_average=None, reduce=None, reduction='mean')
    output_BCE = loss_BCE(input_c, target_c).to(device)
    # loss formula
    if partial_training == 'classification':
        Loss = output_BCE
    elif partial_training == 'regression':
        Loss = output_MSE
    else:
        Loss = output_MSE+output_BCE
    return Loss


# steering type=0
def custom_mse_id_nemo(y_true, y_pred, fc_quantum, device): # Mean Squared Error
    target_s = y_true[:,0].squeeze()
    input_s  = y_pred[0]
    input_s_id = input_s * fc_quantum
    loss_MSE = nn.MSELoss(size_average=None, reduce=None, reduction='mean')
    output_MSE = loss_MSE(input_s_id, target_s).to(device)
    return output_MSE



################################################################################
# Other Utils
################################################################################

def get_fc_quantum(args, model):
    if args.bypass:
        fc_quantum = model.fc.get_output_eps(
            model.Block3.add.get_output_eps(
                model.get_eps_at('Block3.add', eps_in=1./255)
            )
        )
    else:
        fc_quantum = model.fc.get_output_eps(
            model.Block3.relu2.get_output_eps(
                model.get_eps_at('Block3.relu2', eps_in=1./255)
            )
        )
    return fc_quantum

def init_weights(m):
    if type(m) == nn.Linear or type(m) == nn.Conv2d:
        torch.nn.init.xavier_uniform_(m.weight)
        try:
            m.bias.data.fill_(0.01)
        except:
            print('warning: no bias defined in layer', m)


class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=7, verbose=False, delta=0, path='checkpoint.pth'):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement.
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
            path (str): Path for the checkpoint to be saved to.
                            Default: 'checkpoint.pt'
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.path = path

    def __call__(self, val_loss, model):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss

################################################################################
# Visualization
################################################################################

import cv2
# Colors (B, G, R)
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
LIGHT_GREEN = (105, 255, 105)
RED = (0, 0, 255)
BLUE = (200, 0, 0)
LIGHT_BLUE = (188, 234, 254)
GREEN = (0, 200, 0)
GRAY = (100, 100, 100)
def create_cv2_image(img_path, yaw_rate_label, collision_label, yaw_rate_output, collision_output):

        # --- LOAD IMAGE ---
        img = cv2.imread(img_path, 0)
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        #get image size
        height, width = img.shape[0:2]

        # --- image margin from borders ---
        margin = 10 # pixels
        yaw_bar_size = width - 2*margin
        collision_bar_size = height - 2*margin

        # Normalize yaw and collision over image size
        # ground truth
        collision_label = int(collision_label * collision_bar_size)
        yaw_rate_label = yaw_rate_label*90 #optional
        yaw_rate_label = - int(yaw_rate_label)  #minus sign because axis are inverted on opencv

        # network output
        collision_output = int(collision_output * collision_bar_size)
        yaw_rate_output = yaw_rate_output*90 #optional
        yaw_rate_output = - int(yaw_rate_output)  #minus sign because axis are inverted on opencv

        # --- COLLISION ---
        ### GROUND TRUTH BAR ####
        # light blue: EMPTY COLLISION BIN
        img = cv2.line(img,
                        pt1=(margin, margin),    # start_point
                        pt2=(margin, height-margin),   # end_point
                        color= LIGHT_BLUE,
                        thickness= 5)
        # dark blue: 0 = NO COLLISION, 1 COLLISION
        img = cv2.line(img,
                        pt1 =(margin, height - margin - collision_label),    # start_point
                        pt2 =(margin,height-margin),   # end_point
                        color = BLUE,
                        thickness= 5)

        ### OUTPUT BAR ####
        # light blue: EMPTY COLLISION BIN
        img = cv2.line(img,
                        pt1=(margin*2, margin),    # start_point
                        pt2=(margin*2, height-margin),   # end_point
                        color= LIGHT_BLUE,
                        thickness= 5)
        # dark blue: 0 = NO COLLISION, 1 COLLISION
        img = cv2.line(img,
                        pt1 =(margin*2, height - margin - collision_output),    # start_point
                        pt2 =(margin*2,height-margin),   # end_point
                        color = LIGHT_GREEN,
                        thickness= 5)

        # --- YAW RATE ---  light green
        # half circle
        height, width = img.shape[0:2]
        # Ellipse parameters. NOTE: Make sure all the ellipse parameters are int otherwise it raises "TypeError: ellipse() takes at most 5 arguments (8 given)"
        radius = int(40)
        center = (int(width/2), int(height - margin))
        axes = (radius, radius)
        angle = int(0)
        startAngle = int(180)
        endAngle = int(360)
        thickness = int(1)
        img = cv2.ellipse(img, center, axes, angle, startAngle, endAngle, WHITE, thickness)

        # MOVING DOT GROUND TRUTH
        startAngle = int(270+yaw_rate_label)
        endAngle = int(270+yaw_rate_label+1)
        thickness = int(10)
        img = cv2.ellipse(img, center, axes, angle, startAngle, endAngle, BLUE, thickness)

        # MOVING DOT OUTPUT
        startAngle = int(270+yaw_rate_output)
        endAngle = int(270+yaw_rate_output+1)
        thickness = int(10)
        center = (int(width/2), int(height - margin*2))
        img = cv2.ellipse(img, center, axes, angle, startAngle, endAngle, LIGHT_GREEN, thickness)

        return img

################################################################################
# Argument Parsing
################################################################################

import argparse
def str2bool(value):
    """
    Convert a string to a boolean.
    Accepts 'yes', 'true', 't', 'y', '1' as True, and 'no', 'false', 'f', 'n', '0' as False.
    """
    if isinstance(value, bool):
        return value
    if value.lower() in {'yes', 'true', 't', 'y', '1'}:
        return True
    elif value.lower() in {'no', 'false', 'f', 'n', '0'}:
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


################################################################################
# V2 Utils
################################################################################

def label_steering_to_yawrate_V2(labels, types):
    if (types==0).sum().item() > 0:
        # clamp in [-1;+1] range
        labels[types==0] = torch.clamp(labels[types==0], min=-1, max=1)
        # scale to Yaw-Rate Range
        labels[types==0] = labels[types==0] * 90 # deg/s
        # change variable name
        yawrate_labels = labels
        return yawrate_labels
    else:
        return labels

def output_steering_to_yawrate(outputs):
    # clamp in [-1;+1] range
    outputs[0] = torch.clamp(outputs[0], min=-1, max=1)
    # scale to Yaw-Rate Range
    outputs[0] = outputs[0] * 90
    return outputs

def label_steering_to_yawrate_V3(labels):
    # scale to Yaw-Rate Range
    labels[:,0] = labels[:,0] * 90 # deg/s
    # change variable name
    yawrate_labels = labels
    return yawrate_labels


# steering type=0
def custom_mse_v2(y_true, y_pred, t, device): # Mean Squared Error

    output_MSE = torch.zeros(1, 1, dtype=torch.float32, requires_grad=False)

    if (t==0).sum().item() > 0:
        target_s = y_true[t==0]
        input_s  = y_pred[0][t==0]
        loss_MSE = nn.MSELoss(size_average=None, reduce=None, reduction='mean')
        output_MSE = loss_MSE(input_s, target_s).to(device)
        return output_MSE, 1
    else:
        return output_MSE, 0


# collision type=1
def custom_accuracy_v2(y_true, y_pred, t, device):

    tp = 0.0
    tn = 0.0
    fp = 0.0
    fn = 0.0
    acc = torch.zeros(1, 1, dtype=torch.float32, requires_grad=False)
    pre = 0.0
    rec = 0.0
    f1 = 0.0

    if (t==1).sum().item() > 0:
        pred =  y_pred[1][t==1] >= 0.5
        truth = y_true[t==1] >= 0.5
        tp += pred.mul(truth).sum(0).float()
        tn += (~pred).mul(~truth).sum(0).float()
        fp += pred.mul(~truth).sum(0).float()
        fn += (~pred).mul(truth).sum(0).float()
        acc = (tp + tn).sum() / (tp + tn + fp + fn).sum()
        pre = tp / (tp + fp)
        rec = tp / (tp + fn)
        f1 = (2.0 * tp) / (2.0 * tp + fp + fn)
        # avg_pre = nanmean(pre)
        # avg_rec = nanmean(rec)
        # avg_f1 = nanmean(f1)
        return acc, 1
    else:
        return acc, 0


# collision type=1
def custom_bce_v2(y_true, y_pred, t, device): # Debunking loss functions

    output_BCE = torch.zeros(1, 1, dtype=torch.float32, requires_grad=False)

    if (t==1).sum().item() > 0:
        target_c = y_true[t==1]
        input_c  = y_pred[1][t==1]
        loss_BCE = nn.BCELoss(weight=None, size_average=None, reduce=None, reduction='mean')
        output_BCE = loss_BCE(input_c, target_c).to(device)
        return output_BCE, 1
    else:
        return output_BCE, 0


# Labels: steering = 0, collision = 1
def custom_loss_v2(y_true, y_pred, t, epoch, args, device):  #prime 10 epoche traino solo su steering. dal ep 10 inizio a trainare anche su classification

    output_MSE = torch.zeros(1, 1, dtype=torch.float32, requires_grad=True).to(device)
    output_BCE = torch.zeros(1, 1, dtype=torch.float32, requires_grad=True).to(device)

    if args.hard_mining_train:
        global alpha, beta
        batch_size_training = args.batch_size
        k_mse = np.round(batch_size_training-(batch_size_training-10)*(np.maximum(0.0,1.0-np.exp(-1.0/30.0*(epoch-30.0)))))
        k_entropy = np.round(batch_size_training-(batch_size_training-5)*(np.maximum(0.0,1.0-np.exp(-1.0/30.0*(epoch-30.0)))))
        alpha = 1.0
        beta = np.maximum(0.0, 1.0-np.exp(-1.0/10.0*(epoch-10)))

    n_samples_mse = (t==0).sum().item()
    n_samples_entropy = (t==1).sum().item()

    if n_samples_mse > 0:
        k = int(min(k_mse, n_samples_mse)) # k_mse è quello effettivo: prendiamo i top k. calcolo come minimo tra k' (si aggiorna cin modo dinamico) e il minimo delle label tra steering e collision
        target_s = y_true[t==0]
        input_s  = y_pred[0][t==0]
        if args.hard_mining_train:
            # output_MSE = torch.mul((input_s-target_s),(input_s-target_s)).mean()
            l_mse = torch.mul((input_s-target_s),(input_s-target_s))
            output_MSE_value, output_MSE_idx = torch.topk(l_mse, k)
            output_MSE = output_MSE_value.mean().to(device)
        else:
            loss_MSE = nn.MSELoss(size_average=None, reduce=None, reduction='mean')
            output_MSE = loss_MSE(input_s, target_s).to(device)

    if n_samples_entropy > 0:
        k = int(min(k_entropy, n_samples_entropy))
        target_c = y_true[t==1]
        input_c  = y_pred[1][t==1]
        if args.hard_mining_train:
            # output_BCE = F.binary_cross_entropy(input_c, target_c, reduce=False).mean()
            l_bce = F.binary_cross_entropy(input_c, target_c, reduce=False)
            output_BCE_value, output_BCE_idx = torch.topk(l_bce, k)
            output_BCE = output_BCE_value.mean().to(device)
        else:
            loss_BCE = nn.BCELoss(weight=None, size_average=None, reduce=None, reduction='mean')
            output_BCE = loss_BCE(input_c, target_c).to(device)

    if args.hard_mining_train:
        return output_MSE*alpha+output_BCE*beta
    else:
        return output_MSE+output_BCE


# steering type=0
def custom_mse_id_nemo_v2(y_true, y_pred, t, fc_quantum, device): # Mean Squared Error

    output_MSE = torch.zeros(1, 1, dtype=torch.float32, requires_grad=False)

    if (t==0).sum().item() > 0:
        target_s = y_true[t==0]
        input_s  = y_pred[0][t==0]
        input_s_id =input_s * fc_quantum
        loss_MSE = nn.MSELoss(size_average=None, reduce=None, reduction='mean')
        output_MSE = loss_MSE(input_s_id, target_s).to(device)
        return output_MSE, 1
    else:
        return output_MSE, 0
