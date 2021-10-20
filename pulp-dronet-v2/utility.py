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
# File:   utility.py                                                            #
# Author: Daniele Palossi  <dpalossi@iis.ee.ethz.ch> <daniele.palossi@idsia.ch> #
#         Lorenzo Lamberti <lorenzo.lamberti@unibo.it>                          #
#         Vlad Niculescu   <vladn@iis.ee.ethz.ch>                               #
# Date:   18.02.2021                                                            #
#-------------------------------------------------------------------------------#

# essentials
import os
from os.path import join
import numpy as np
from PIL import Image
# torch
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset


################################################################################
# Dataset Loader
################################################################################

# steering
def read_csv_label(filename):
    try:
        return np.loadtxt(filename, usecols=6, delimiter=',', skiprows=0)
    except OSError as e:
         print("No labels found in dir", filename)

# collision
def read_txt_label(filename):
    try:
        return np.loadtxt(filename, usecols=0)
    except OSError as e:
         print("No labels found in dir", filename)

class DronetDataset(Dataset):
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
                        l_steering = read_csv_label(os.path.join(full_dir_path, file))
                        self.labels.extend(l_steering)
                        self.types.extend([0]*len(l_steering))
                    if file.endswith(".txt"):
                        l_collision = read_txt_label(os.path.join(full_dir_path, file))
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
# Loss Functions and Evaluation Metrics
################################################################################

# Label indexes: steering = 0
def custom_mse(y_true, y_pred, t, device): # Mean Squared Error

    output_MSE = torch.zeros(1, 1, dtype=torch.float32, requires_grad=False)
    
    if (t==0).sum().item() > 0:
        target_s = y_true[t==0]
        input_s  = y_pred[0][t==0]
        loss_MSE = nn.MSELoss(size_average=None, reduce=None, reduction='mean')
        output_MSE = loss_MSE(input_s, target_s).to(device)
        return output_MSE, 1
    else:
        return output_MSE, 0


# Label indexes: collision = 1
def custom_accuracy(y_true, y_pred, t, device):

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


# Label indexes: collision = 1
def custom_bce(y_true, y_pred, t, device): # Debunking loss functions

    output_BCE = torch.zeros(1, 1, dtype=torch.float32, requires_grad=False)
    
    if (t==1).sum().item() > 0:
        target_c = y_true[t==1]
        input_c  = y_pred[1][t==1]
        loss_BCE = nn.BCELoss(weight=None, size_average=None, reduce=None, reduction='mean')
        output_BCE = loss_BCE(input_c, target_c).to(device)
        return output_BCE, 1
    else:
        return output_BCE, 0


# Label indexes: steering = 0, collision = 1
def custom_loss(y_true, y_pred, t, epoch, args, device):  #prime 10 epoche traino solo su steering. dal ep 10 inizio a trainare anche su classification

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


# Label indexes: steering = 0
def custom_mse_id_nemo(y_true, y_pred, t, fc_quantum, device): # Mean Squared Error

    output_MSE = torch.zeros(1, 1, dtype=torch.float32, requires_grad=False)
    
    if (t==0).sum().item() > 0:
        target_s = y_true[t==0]
        input_s  = y_pred[0][t==0]
        input_s_id = input_s * fc_quantum
        loss_MSE = nn.MSELoss(size_average=None, reduce=None, reduction='mean')
        output_MSE = loss_MSE(input_s_id, target_s).to(device)
        return output_MSE, 1
    else:
        return output_MSE, 0


################################################################################
# Other Utils
################################################################################

# nemo util
def get_fc_quantum(model):
    fc_quantum = model.fc.get_output_eps(model.resBlock3.add.get_output_eps(model.get_eps_at('resBlock3.add', eps_in=1./255)))
    return fc_quantum

# initialization of weights
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
