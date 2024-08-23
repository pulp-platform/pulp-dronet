#-----------------------------------------------------------------------------#
# Copyright(C) 2024 University of Bologna, Italy, ETH Zurich, Switzerland.    #
# All rights reserved.                                                        #
#                                                                             #
# Licensed under the Apache License, Version 2.0 (the "License");             #
# you may not use this file except in compliance with the License.            #
# See LICENSE in the top directory for details.                     #
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
# File:    DatasetLogger.py                                                   #
# Authors:                                                                    #
#          Lorenzo Lamberti <lorenzo.lamberti@unibo.it>                       #
#          Daniel Rieben    <riebend@student.ethz.ch>                 #
# Date:    01.03.2024                                                         #
#-----------------------------------------------------------------------------#

from cflib.crazyflie.log import LogConfig, LogVariable, LogTocElement
import logging
from typing import List

class DatasetLogger:
    def __init__(self, log_configs: List[LogConfig], name="DatasetLogger"):
        super().__init__()
        self.logger = logging.getLogger(name)
        self.streamHandler = logging.StreamHandler()
        self.streamHandler.setLevel(logging.INFO)
        format_str = ""
        for config in log_configs:
            var : LogVariable
            for var in config.variables:
                type = LogTocElement.get_cstring_from_id(var.fetch_as)
                print(var)

        self.formatter = logging.Formatter()


if __name__ == "__main__":
    lg_stab = LogConfig(name='Stabilizer', period_in_ms=500)  # Log.c changed s.t. 500 = 50ms
    lg_stab.add_variable('stabilizer.roll', 'float')
    lg_stab.add_variable('stabilizer.pitch', 'float')
    lg_stab.add_variable('stabilizer.yaw', 'float')
    lg_stab.add_variable('stabilizer.thrust', 'float')
    lg_stab.add_variable('range.front', 'uint16_t')
    logger = DatasetLogger([lg_stab])
