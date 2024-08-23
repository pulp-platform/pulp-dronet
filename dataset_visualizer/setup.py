#-----------------------------------------------------------------------------#
# Copyright(C) 2024 University of Bologna, Italy, ETH Zurich, Switzerland.    #
# All rights reserved.                                                        #
#                                                                             #
# Licensed under the Apache License, Version 2.0 (the "License");             #
# you may not use this file except in compliance with the License.            #
# See LICENSE in the top directory for details.                               #
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
# File:    setup.py                                                           #
# Authors:                                                                    #
#          Michal Barcis    <michal.barcis@tii.ae>                            #
#          Lorenzo Lamberti <lorenzo.lamberti@unibo.it>                       #
# Date:    01.03.2024                                                         #
#-----------------------------------------------------------------------------#

from distutils.core import setup

setup(name='dataset_visualizer',
    version='1.0',
    description='A Python package used to visualize the dataset of the PULP-DroNet project',
    packages=['dataset_visualizer'],
    install_requires=[
        'opencv-python>=4.5.3.56',
        'tk>=0.1.0',
        'pandas>=1.2.3',
        'pillow>=8.1.2',
    ],
)