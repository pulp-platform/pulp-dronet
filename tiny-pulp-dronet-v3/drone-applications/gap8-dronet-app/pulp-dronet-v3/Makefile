# Makefile
# Alessio Burrello <alessio.burrello@unibo.it>
#
# Copyright (C) 2019-2020 University of Bologna
# 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

APP = main
APP_SRCS = DORY_network/src/pulp_nn_depthwise_generic.c

APP_SRCS+=DORY_network/src/pulp_nn_conv_HoWo_parallel.c
APP_SRCS+=DORY_network/src/layerConvBNRelu7.c
APP_SRCS+=DORY_network/src/layerConvBNRelu3.c
APP_SRCS+=DORY_network/src/pulp_nn_avgpool.c
APP_SRCS+=DORY_network/src/network.c
APP_SRCS+=DORY_network/src/layerConvBNRelu2.c
APP_SRCS+=DORY_network/src/pulp_nn_maxpool.c
APP_SRCS+=DORY_network/src/layerConvBNRelu8.c
APP_SRCS+=DORY_network/src/pulp_nn_depthwise_generic_less_4_weights.c
APP_SRCS+=DORY_network/src/pulp_nn_conv_Co_parallel.c
APP_SRCS+=DORY_network/src/layerConvBNRelu12.c
APP_SRCS+=DORY_network/src/dory.c
APP_SRCS+=DORY_network/src/layerMaxPool1.c
APP_SRCS+=DORY_network/src/layerConvBNRelu4.c
APP_SRCS+=DORY_network/src/pulp_nn_pointwise_Ho_parallel.c
APP_SRCS+=DORY_network/src/layerAddRelu9.c
APP_SRCS+=DORY_network/src/pulp_nn_utils.c
APP_SRCS+=DORY_network/src/pulp_nn_conv_Ho_parallel.c
APP_SRCS+=DORY_network/src/layerConvBNRelu0.c
APP_SRCS+=DORY_network/src/layerAddRelu5.c
APP_SRCS+=DORY_network/src/pulp_nn_linear_out_32.c
APP_SRCS+=DORY_network/src/pulp_nn_linear.c
APP_SRCS+=DORY_network/src/layerConvBNRelu6.c
APP_SRCS+=DORY_network/src/layerMatMul14_last.c
APP_SRCS+=DORY_network/src/pulp_nn_depthwise_3x3_s1.c
APP_SRCS+=DORY_network/src/pulp_nn_add.c
APP_SRCS+=DORY_network/src/pulp_nn_matmul.c
APP_SRCS+=DORY_network/src/mem_controller.c
APP_SRCS+=DORY_network/src/layerConvBNRelu11.c
APP_SRCS+=DORY_network/src/layerConvBNRelu10.c
APP_SRCS+=DORY_network/src/pulp_nn_pointwise_Co_parallel.c
APP_SRCS+=DORY_network/src/layerAddRelu13.c
APP_SRCS+=DORY_network/src/pulp_nn_pointwise_HoWo_parallel.c
APP_SRCS+=DORY_network/src/main.c

# Added
APP_SRCS+=$(GAP_SDK_HOME)/libs/gap_lib/img_io/ImgIO.c

APP_LDFLAGS += -lgaptools -lgaplib

ifndef CORE
CORE=8
else
CORE= $(CORE)
endif

APP_CFLAGS += -DNUM_CORES=$(CORE) -DGAP_SDK=1 -IDORY_network/inc -O3 -w -fno-tree-loop-distribute-patterns -flto
APP_LDFLAGS += -lm -flto


FLASH_FILES = DORY_network/ConvBNRelu0_weights.hex
FLASH_FILES += DORY_network/ConvBNRelu2_weights.hex
FLASH_FILES += DORY_network/ConvBNRelu3_weights.hex
FLASH_FILES += DORY_network/ConvBNRelu4_weights.hex
FLASH_FILES += DORY_network/ConvBNRelu6_weights.hex
FLASH_FILES += DORY_network/ConvBNRelu7_weights.hex
FLASH_FILES += DORY_network/ConvBNRelu8_weights.hex
FLASH_FILES += DORY_network/ConvBNRelu10_weights.hex
FLASH_FILES += DORY_network/ConvBNRelu11_weights.hex
FLASH_FILES += DORY_network/ConvBNRelu12_weights.hex
FLASH_FILES += DORY_network/MatMul14_weights.hex
FLASH_FILES += DORY_network/inputs.hex
READFS_FILES := $(FLASH_FILES)
PLPBRIDGE_FLAGS += -f

include $(RULES_DIR)/pmsis_rules.mk
