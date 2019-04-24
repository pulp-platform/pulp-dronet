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
# File:    run_dataset.sh                                                      #
# Author:  Daniele Palossi <dpalossi@iis.ee.ethz.ch>                           #
# Date:    10.04.2019                                                          #
#------------------------------------------------------------------------------#


#!/bin/bash

################################ Entire Dataset ################################
BASEDIR=$(pwd)/../dataset/

################################### Sub-Sets ###################################
# BASEDIR=$(pwd)/../dataset/Himax_Dataset/
# BASEDIR=$(pwd)/../dataset/Udacity_Dataset/
# BASEDIR=$(pwd)/../dataset/Zurich_Bicycle_Dataset/

################################## Sub-Folders #################################
# BASEDIR=$(pwd)/../dataset/Himax_Dataset/test_2/
# BASEDIR=$(pwd)/../dataset/Himax_Dataset/test_10/
# BASEDIR=$(pwd)/../dataset/Himax_Dataset/test_15/
# BASEDIR=$(pwd)/../dataset/Himax_Dataset/test_23/


for s in $(find $BASEDIR -name "*.pgm" | sort -V); do
	# echo "Testing: $s"
	make conf run EXT_INPUT=$s | grep Result
done