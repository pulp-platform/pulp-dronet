#  Copyright (C) 2024 University of Bologna, Italy, ETH Zurich, Switzerland.
#  All rights reserved.

#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  See LICENSE.apache.md in the top directory for details.
#  You may obtain a copy of the License at

#    http://www.apache.org/licenses/LICENSE-2.0

#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.

#  File:    Makefile
#  Author:  Lorenzo Lamberti      <lorenzo.lamberti@unibo.it>
#  Date:    01.03.2024

APP = wifi_frame_streamer
APP_SRCS += wifi_frame_streamer.c

APP_CFLAGS += -O3 -g

APP_LDFLAGS += -lgaptools -lgaplib -lm

RUNNER_CONFIG = $(CURDIR)/config.ini

APP_SRCS := $(filter-out $(GAP_SDK_HOME)/tools/gap_tools/frame_streamer/frame_streamer.c, $(APP_SRCS))
APP_INC := $(filter-out $(GAP_SDK_HOME)/home/bitcraze/gap_sdk/tools/gap_tools/include/tools, $(APP_INC))
APP_INC := $(filter-out $(wildcard *frame_streamer.h), $(APP_INC))

include $(RULES_DIR)/pmsis_rules.mk
