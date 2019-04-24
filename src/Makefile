#------------------------------------------------------------------------------#
# Copyright (C) 2018-2019 ETH Zurich, Switzerland                              #
# All rights reserved.                                                         #
#                                                                              #
# Licensed under the Apache License, Version 2.0 (the "License");              #
# you may not use this file except in compliance with the License.             #
# See LICENSE.apache.md in the top directory for details.                      #
# You may obtain a copy of the License at                                      #
#                                                                              #
#     http://www.apache.org/licenses/LICENSE-2.0                               #
#                                                                              #
# Unless required by applicable law or agreed to in writing, software          #
# distributed under the License is distributed on an "AS IS" BASIS,            #
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.     #
# See the License for the specific language governing permissions and          #
# limitations under the License.                                               #
#                                                                              #
# File:    Makefile                                                            #
# Author:  Daniele Palossi <dpalossi@iis.ee.ethz.ch>                           #
# Date:    10.04.2019                                                          #
#------------------------------------------------------------------------------#


pulpChip			= GAP
PULP_APP			= PULPDroNet
WEIGHTS_DIR			= $(CURDIR)/../weights
TILER_DIR			= $(CURDIR)/autotiler
PULP_APP_SRCS		= PULPDronet.c PULPDronetKernels.c PULPDronetKernelsInit.c $(TILER_DIR)/src/CNN_BasicKernels.c
pulpFc				= 1
RM					= rm -rf
PULP_ARCH_CFLAGS	= -mPE=8 -mFC=1
PULP_ARCH_LDFLAGS	= -mPE=8 -mFC=1
PULP_CFLAGS			+= -O3 -w -mno-memcpy -fno-tree-loop-distribute-patterns -fdata-sections -ffunction-sections -Wall -Wno-maybe-uninitialized -Wno-unused-but-set-variable
PULP_CFLAGS 		+= -I$(TILER_DIR)/include -I$(CURDIR)
PULP_LDFLAGS		+= -flto -Wl,--gc-sections
USER_CONFIG			= $(CURDIR)/config.ini


all::

autotiler: clean_tiler generate

build_tiler:
	gcc -o AutoTiler -I$(TILER_DIR)/include -I$(CURDIR) PULPDronetGenerator.c $(TILER_DIR)/src/CNN_Generator.c $(TILER_DIR)/lib/libtile.a
	
generate: build_tiler
	./AutoTiler -K -O

clean::
	$(RM) build

clean_tiler: clean
	$(RM) AutoTiler PULPDronetKernels.* PULPDronetKernelsInit.*

.PHONY: generate clean


################################### WEIGHTS ####################################
override CONFIG_OPT += 	flash/fs/files=$(WEIGHTS_DIR)/binary/weights_conv2d_1.hex \
						flash/fs/files=$(WEIGHTS_DIR)/binary/weights_conv2d_2.hex \
						flash/fs/files=$(WEIGHTS_DIR)/binary/weights_conv2d_3.hex \
						flash/fs/files=$(WEIGHTS_DIR)/binary/weights_conv2d_4.hex \
						flash/fs/files=$(WEIGHTS_DIR)/binary/weights_conv2d_5.hex \
						flash/fs/files=$(WEIGHTS_DIR)/binary/weights_conv2d_6.hex \
						flash/fs/files=$(WEIGHTS_DIR)/binary/weights_conv2d_7.hex \
						flash/fs/files=$(WEIGHTS_DIR)/binary/weights_conv2d_8.hex \
						flash/fs/files=$(WEIGHTS_DIR)/binary/weights_conv2d_9.hex \
						flash/fs/files=$(WEIGHTS_DIR)/binary/weights_conv2d_10.hex \
						flash/fs/files=$(WEIGHTS_DIR)/binary/weights_dense_1.hex \
						flash/fs/files=$(WEIGHTS_DIR)/binary/weights_dense_2.hex 


#################################### BIASES ####################################
override CONFIG_OPT +=	flash/fs/files=$(WEIGHTS_DIR)/binary/bias_conv2d_1.hex \
						flash/fs/files=$(WEIGHTS_DIR)/binary/bias_conv2d_2.hex \
						flash/fs/files=$(WEIGHTS_DIR)/binary/bias_conv2d_3.hex \
						flash/fs/files=$(WEIGHTS_DIR)/binary/bias_conv2d_4.hex \
						flash/fs/files=$(WEIGHTS_DIR)/binary/bias_conv2d_5.hex \
						flash/fs/files=$(WEIGHTS_DIR)/binary/bias_conv2d_6.hex \
						flash/fs/files=$(WEIGHTS_DIR)/binary/bias_conv2d_7.hex \
						flash/fs/files=$(WEIGHTS_DIR)/binary/bias_conv2d_8.hex \
						flash/fs/files=$(WEIGHTS_DIR)/binary/bias_conv2d_9.hex \
						flash/fs/files=$(WEIGHTS_DIR)/binary/bias_conv2d_10.hex \
						flash/fs/files=$(WEIGHTS_DIR)/binary/bias_dense_1.hex \
						flash/fs/files=$(WEIGHTS_DIR)/binary/bias_dense_2.hex


################################# CAMERA IMGS ##################################

# Use this if you want to test a single image
override CONFIG_OPT += camera/image-stream=$(CURDIR)/../dataset/Himax_Dataset/test_2/frame_22.pgm

# Use this if you want to test the dataset
# override CONFIG_OPT += camera/image-stream=$(EXT_INPUT)


include $(PULP_SDK_HOME)/install/rules/pulp_rt.mk