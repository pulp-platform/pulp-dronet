# Copyright (C) 2020-2021 ETH Zurich, Switzerland, University of Bologna, Italy.
# All rights reserved.                                                           
                                                                            
# Licensed under the Apache License, Version 2.0 (the "License");               
# you may not use this file except in compliance with the License.              
# See LICENSE.apache.md in the top directory for details.                       
# You may obtain a copy of the License at                                       
                                                                            
#   http://www.apache.org/licenses/LICENSE-2.0                                  
                                                                            
# Unless required by applicable law or agreed to in writing, software           
# distributed under the License is distributed on an "AS IS" BASIS,             
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.      
# See the License for the specific language governing permissions and           
# limitations under the License.                                                
                                                                            
# File:    nntool_model_eval.py      
# Author:  Vlad Niculescu      <vladn@iis.ee.ethz.ch>                           
# Date:    15.03.2021                                                           

import os
import sys
sys.path.append('../')
from os.path import join 

import argparse
import logging
import numpy as np
from cmd2 import Cmd, Cmd2ArgumentParser, with_argparser

from execution.graph_executer import GraphExecuter
from execution.quantization_mode import QuantizationMode
from execution.execution_progress import ExecutionProgress
from interpreter.nntool_shell_base import NNToolShellBase
from interpreter.shell_utils import (glob_input_files, input_options)

from utils.data_importer import import_data
from sklearn.metrics import mean_squared_error, explained_variance_score
from glob import glob


def create_parser(cfg):
    parser = argparse.ArgumentParser(description='PyTorch PULP-DroNet Testing')
    parser.add_argument('-d', '--data_path', help='path to dataset',
                        default=cfg.data_path)
    return parser

def _get_input_args():
    res = {}
    res['width'] = res['height'] = -1
    res['mode'] = None

    res['divisor'] = 1
    res['offset'] = 0
    res['transpose'] = False
    res['norm_func'] = ''

    return res

def glob_input_files2(input_files, graph_inputs=1):
    input_files_list = []
    for file in input_files:
        for globbed_file in glob(file):
            input_files_list.append(globbed_file)
    if len(input_files_list) % graph_inputs:
        return ValueError("input files number is not divisible for graph inputs {}".format(graph_inputs))
    shard = int(len(input_files_list) / graph_inputs)
    return [[input_files_list[i+j] for i in range(0, len(input_files_list), shard)] \
                for j in range(shard)]

def sigmoid(z):
    return 1/(1 + np.exp(-z))

def get_labels(folder_path):
    files_in_folder = os.listdir(folder_path)
    for subfile in files_in_folder:
        if subfile.endswith(".txt"):
            labels = np.genfromtxt(folder_path + "/" + subfile, delimiter=",")
            return labels, 0

        if subfile.endswith(".csv"):
            labels = np.loadtxt(folder_path + "/" + subfile, usecols=6, delimiter=',', skiprows=0)
            return labels, 1

    print("Error: No labels file found!")
    return -1, -1

def custom_mse_evs(y_true, y_pred):
    y_true_np = np.array(y_true).reshape(-1)
    y_pred_np = np.array(y_pred).reshape(-1)
    mse = len(y_true_np) * mean_squared_error(y_true_np, y_pred_np)
    evs = len(y_true_np) * explained_variance_score(y_true_np, y_pred_np)
    return mse, evs, len(y_true_np)

def custom_accuracy(y_true, y_pred):
    y_true_np = np.array(y_true).reshape(-1).astype(int)
    y_pred_np = np.array(y_pred).reshape(-1)

    y_pred_bin = []
    for i in range(0, len(y_true_np)):
        if y_pred_np[i] >= 0.5:
            y_pred_bin.append(1)
        else:
            y_pred_bin.append(0)
    y_pred_np_bin = np.array(y_pred_bin)

    tp = np.multiply(y_pred_np_bin, y_true_np).sum()
    tn = np.multiply(np.bitwise_not(y_pred_np_bin) + 2, np.bitwise_not(y_true_np) + 2).sum()
    fp = np.multiply(y_pred_np_bin, ~y_true_np+2).sum()
    fn = np.multiply(~y_pred_np_bin + 2, y_true_np).sum()
    acc = (tp + tn) / (tp + tn + fp + fn)

    return acc * len(y_true), len(y_true), tp, tn, fp, fn


def validate(INPUT):
    input_args =_get_input_args()

    good_predictions = []
    good_margin = 0
    bad_margin = 0
    qmode = QuantizationMode.all_dequantize()
    # qmode = QuantizationMode.none()

    accuracy_list = []
    mse_list = []
    evs_list = []
    acc_samples = 0
    reg_samples = 0
    tp_total, tn_total, fp_total, fn_total = 0.0, 0.0, 0.0, 0.0

    folders_list = os.listdir(INPUT)
    folders_list.sort()
    file_counter = 0
    logged_data_reg = np.array([0, 0, 0, 0]).reshape(-1, 4)
    logged_data_acc = np.array([0, 0, 0, 0]).reshape(-1, 4)
    for folder in folders_list:
        labels, is_regression = get_labels(INPUT + folder)
        try:
            files_list = glob_input_files2([INPUT + folder + "/images/*"], G.num_inputs)
            files_list.sort()
            predictions_0 = []
            predictions_1 = []
            for i, file_per_input in enumerate(files_list):
                data = [import_data(input_file, **input_args) for input_file in file_per_input]
                data = [x / 255 for x in data] # scale data
                executer = GraphExecuter(G, qrecs=G.quantization)
                outputs = executer.execute(data, qmode=qmode)

                pred0 = np.asarray(outputs[40])
                pred0 = pred0[0][0]
                pred1 = np.asarray(outputs[42])
                pred1 = pred1[0][0]

                print("File nr. ", file_counter)
                file_counter += 1
                # print("File: ", file_per_input, ":  ", pred0, pred1)

                predictions_0.append(pred0)
                predictions_1.append(pred1)

                log_entry = np.array([file_per_input[0], pred0, pred1, labels[i]]).reshape(-1, 4)
                if is_regression:
                    logged_data_reg = np.concatenate((logged_data_reg, log_entry), axis=0)
                else:
                    logged_data_acc = np.concatenate((logged_data_acc, log_entry), axis=0)

            if is_regression:
                mse, evs, mse_len = custom_mse_evs(labels, predictions_0)
                reg_samples += mse_len
                mse_list.append(mse)
            else:
                acc, acc_len, tp, tn, fp, fn = custom_accuracy(labels, predictions_1)
                acc_samples += acc_len
                accuracy_list.append(acc)
                tp_total += tp
                tn_total += tn
                fp_total += fp
                fn_total += fn

        except (KeyboardInterrupt, SystemExit):
            pass
    print(" ")
    print("Scores for ", INPUT)
    if reg_samples != 0:
        print("MSE: ", round(sum(mse_list) / reg_samples, 5))
        print("EVS: ", round(sum(evs_list) / reg_samples, 5))

    if acc_samples != 0:
        f1 = (2.0 * tp_total) / (2.0 * tp_total + fp_total + fn_total)
        print("Accuracy: ", round(sum(accuracy_list) / acc_samples, 3))
        print("F1: ", round(f1, 3))



def main():
    # parse arguments
    global args
    from config import cfg # load configuration with all default values 
    parser = create_parser(cfg)
    args = parser.parse_args()
    dataset_path = join('..', args.data_path)

    #validation of the quantized model over the original dataset
    dataset_original = dataset_path + "himax/jpg/testing/"
    validate(dataset_original)

    #validation of the quantized model over the himax dataset
    himax_dataset = dataset_path + "testing/"
    validate(himax_dataset)

if __name__ == '__main__':
    main()

