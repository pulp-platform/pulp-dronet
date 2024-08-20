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
# File:    config.py                                                          #
# Authors:                                                                    #
#          Lorenzo Lamberti <lorenzo.lamberti@unibo.it>                       #
#          Daniele Palossi  <dpalossi@iis.ee.ethz.ch>                         #
# Date:    01.03.2024                                                         #
#-----------------------------------------------------------------------------#

# This file gathers all the default values for the following scripts:
# training.py, testing.py, evaluation.py, quantize.py

class cfg:
    pass

# == For all scripts ==
# dataset
cfg.data_path= './dataset/training/'
cfg.data_path_testing = './dataset/testing/'
cfg.logs_dir='./logs/'
cfg.bypass=True
cfg.depth_mult=1.0
cfg.testing_dataset='original'
cfg.model_weights_path='model/dronet_v3.pth'
cfg.gpu='0'
cfg.workers=4

# training.py
cfg.training_dataset = 'original_and_himax'
cfg.model_name = 'pulp_dronet_v3'
cfg.training_batch_size=32
cfg.epochs=100
cfg.learning_rate = 1e-3
cfg.lr_decay = 1e-5
cfg.checkpoint_path = './checkpoints/'
cfg.hard_mining_train = False
cfg.early_stopping = False
cfg.patience = 15
cfg.delta = 0
cfg.resume_training = False

# testing.py
cfg.testing_batch_size=32

# quantize.py
cfg.nemo_export_path = 'nemo_output/'
cfg.nemo_onnx_name = 'pulp_dronet_id_4dory.onnx'