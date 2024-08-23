#  Copyright (C) 2024 University of Bologna, Italy, ETH Zurich, Switzerland.
#  All rights reserved.
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  See LICENSE in the top directory for details.
#  You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
#
#  File:    Makefile
#  Author:  Lorenzo Lamberti      <lorenzo.lamberti@unibo.it>
#           Daniel Rieben         <riebend@student.ethz.ch>
#  Date:    01.03.2024

# Add folder to the path
CRAZYFLIE_BASE = ../../external/crazyflie-firmware
VPATH += $(CRAZYFLIE_BASE)/../../crazyflie_firmware_patches/dataset_collection_app

# Load the drivers/files used for the data set collector (if you want to load more than one deck change deck_info.c)
CFLAGS += -DDECK_FORCE=datasetCollectorDriver
PROJ_OBJ += datasetcollector.o

DEBUG=1

# Some firmware changes are required so the original files from the crazyflie-firmware are excluded
# and the modified ones are added (add one line for each)
INCLUDES := $(filter-out -I$(CRAZYFLIE_BASE)/src/modules/interface/log.h, $(INCLUDES))
SRC_FILES := $(filter-out $(CRAZYFLIE_BASE)/src/modules/src/log.c, $(SRC_FILES))
# Add the modified files
VPATH += $(CRAZYFLIE_BASE)/../../crazyflie_firmware_patches/crazyflie-firmware-modified-dataset

include $(CRAZYFLIE_BASE)/Makefile
