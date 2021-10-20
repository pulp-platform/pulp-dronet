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
# File:   config.py                                                             #
# Author: Lorenzo Lamberti <lorenzo.lamberti@unibo.it>                          #
#         Daniele Palossi  <dpalossi@iis.ee.ethz.ch> <daniele.palossi@idsia.ch> #
#         Vlad Niculescu   <vladn@iis.ee.ethz.ch>                               #
# Date:   18.02.2021                                                            #
#-------------------------------------------------------------------------------#

# This file gathers all the default values for the following scripts:
# training.py, testing.py, evaluation.py, quantize.py, onnx_converter.py

class cfg:
    pass

# default for all scripts
cfg.data_path= '../pulp-dronet-dataset/'
cfg.logs_path='./logs/'
cfg.flow='nemo_dory'
cfg.testing_dataset='original'
cfg.model_weights='model/dronet_v2_nemo_dory_original.pth'
cfg.gpu='0'
cfg.workers=4

# training.py
cfg.training_dataset = 'original_and_himax'
cfg.model_name = 'pulp_dronet_v2'
cfg.training_batch_size=32
cfg.epochs=100
cfg.learning_rate = 1e-3
cfg.lr_decay = 1e-5
cfg.checkpoint_path = './checkpoints/'
cfg.hard_mining_train = True
cfg.early_stopping = False
cfg.patience = 15
cfg.delta = 0
cfg.resume_training = False

# testing.py
cfg.testing_batch_size=32

# evaluation.py
cfg.testing_dataset_evaluation='validation'
cfg.cherry_picking_path='./checkpoints/pulp_dronet_v2/'

### NEMO/DORY flow: ###
# quantize.py
cfg.nemo_export_path = 'nemo_output/'
cfg.nemo_onnx_name = 'pulp_dronet_id_4dory.onnx'

### GAPflow:  ###
# onnx_converter.py
cfg.gapflow_onnx_export_path = './nntool_input/models_onnx/'
cfg.gapflow_model_weights_original = './model/dronet_v2_gapflow_original.pth'
cfg.gapflow_model_weights_original_himax = './model/dronet_v2_gapflow_original_himax.pth'