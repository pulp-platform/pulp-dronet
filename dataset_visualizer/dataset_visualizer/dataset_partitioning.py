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
# File:    dataset_partitioning.py                                            #
# Authors:                                                                    #
#          Lorenzo Lamberti <lorenzo.lamberti@unibo.it>                       #
#          Lorenzo Bellone  <lorenzo.bellone@tii.ae>                          #
# Date:    01.03.2024                                                         #
#-----------------------------------------------------------------------------#

import argparse
import os
from models import Dataset
from os.path import join
import pandas as pd
from random import shuffle, random


def create_parser():
    parser = argparse.ArgumentParser(
        description=(
            'Dataset partitioning for the dataset collector framework'
        )
    )

    parser.add_argument(
        '-d',
        '--dataset_path',
        help='path to dataset folder ',
        default=os.path.join(os.path.dirname(__file__), '..', 'dataset'))
    parser.add_argument(
        "--random",
        action="store_true")
    return parser


def main():
    parser = create_parser()
    args = parser.parse_args()

    d = Dataset(args.dataset_path)
    d.initialize_from_filesystem()

    for acquisition in d.acquisitions:
        output = [[], [], [], [], [], []]  # no collision or regression, regression, only collision
        for image in acquisition.images:
            if image.labels["yaw_rate"] == 0 and image.labels["collision"] == 0:
                output[0].append(image.name)
            elif image.labels["yaw_rate"] > 0 and image.labels["collision"] == 0:
                output[1].append(image.name)
            elif image.labels["yaw_rate"] > 0 and image.labels["collision"] == 1:
                output[2].append(image.name)
            elif image.labels["yaw_rate"] < 0 and image.labels["collision"] == 0:
                output[3].append(image.name)
            elif image.labels["yaw_rate"] < 0 and image.labels["collision"] == 1:
                output[4].append(image.name)
            else: # image.labels["yaw_rate"] == 0 and image.labels["collision"] == 1:
                output[5].append(image.name)

        if args.random:
            for sample in output:
                shuffle(sample)
        # division in training (70%),  validation (10%) and testing (20%) for each acquisition
        split = [[], [], [], [], [], []]
        for i in range(len(output)):
            for j in range(len(output[i])):
                if   j < 0.7*(len(output[i])-1): split[i].append("train")
                elif j > 0.8*(len(output[i])-1):
                    if (i == 0) or (i == 5):
                        eps = random()
                        if eps >= 0.85:
                            split[i].append("test")
                        else:
                            split[i].append("None")
                    else:
                        split[i].append("test")
                else: split[i].append("valid")

        # Save the labels
        final = {"filename": [], "partition": [], "label_yaw_rate": [], "label_collision": []}
        for image in acquisition.images:
            final["filename"].append(image.name)
            final["label_yaw_rate"].append(image.labels["yaw_rate"])
            final["label_collision"].append(image.labels["collision"])
            for i in range(len(output)):
                if image.name in output[i]:
                    final["partition"].append(split[i][output[i].index(image.name)])

        # Save the csv files in the correspondent acquisition folder:
        if args.random:
            pd.DataFrame.from_dict(final).to_csv(join(acquisition.path,"labels_partitioned_random.csv"), index=False)
        else:
            pd.DataFrame.from_dict(final).to_csv(join(acquisition.path,"labels_partitioned.csv"), index=False)


if __name__ == "__main__":
    main()
