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
# File:    dataset_post_processing.py                                         #
# Authors:                                                                    #
#          Lorenzo Lamberti <lorenzo.lamberti@unibo.it>                       #
# Date:    01.03.2024                                                         #
#-----------------------------------------------------------------------------#

import pandas as pd
import os
from os.path import join
import numpy as np
import glob
import argparse

def create_parser():
    parser = argparse.ArgumentParser(description='Image viewer for the dataset collector framework of the Bitcraze Crazyflie + AI-Deck')
    parser.add_argument('-d', '--data_path', help='path to dataset acquisition# in the ../dataset/ folder',
                        default='acquisition5')
    # parser.add_argument('-c', '--collision_distance', help='set the collision distance in mm', type=int,
    #                     default=1000)

    return parser

def label_images_gap_ticks(dataset_path):
    # This function matches the drone's states and images by acquisition timestamp.
    # We have 2 inputs:
    #   - state_labels*.csv file  contains all the drone's state logged (acquisition_timestamp, roll, pitch, yaw, front_distance). Logging rate ~100 state/s
    #   - images/ folder: contains all the images saved with the GAP8 camera (and WiFi tx). Each image is named with its acquisition timestamp. Logging rate ~10 images/s
    # Therefore: we normally collect the drone's state with an higher logging rate w.r.t. the image logging rate. We will end with more states than images.
    # This function: finds the correspondence between images and states. We only keep 1 state for each image, and we discard the extra states we collected
    # Output: a single csv file that contains (image_timestap.jpeg , state_timestamp , roll , pitch , yaw , thrust , range.front , mRange.rangeStatusFront)
    state_labels_files = glob.glob(os.path.join(dataset_path, "state_labels*.csv"))
    def key(image_name):
        return int(image_name.split('.')[0])
    image_names = sorted(os.listdir(os.path.join(dataset_path, "images")), key=key)
    image_ticks = [int(img_name.split('.')[0]) for img_name in image_names]
    labeled_images_df = pd.DataFrame(image_names, columns=["timeTicks.jpeg"])
    for state_labels_file in state_labels_files:
        state_labels = pd.read_csv(state_labels_file)
        if "ticks" in state_labels.columns.values:
            state_labels = state_labels.rename(columns={"ticks": "timeTicks"})
            print(state_labels.columns)
        minimum_distance_idxs = []
        for image_tick in image_ticks:
            minimum_distance_idxs.append(np.argmin(abs(state_labels["timeTicks"].values - image_tick)))
        close_state_samples = state_labels.iloc[minimum_distance_idxs, :].reset_index(drop=True)
        confName = state_labels_file.split("_")[-1].split('.')[0]
        close_state_samples = close_state_samples.rename(columns={"timeTicks": confName + "_TimeTicks"})
        labeled_images_df = pd.concat([labeled_images_df, close_state_samples], axis=1)
    return labeled_images_df


def match_images_and_states(acquisition_path):
    # This function matches the drone's states and images by acquisition
    # timestamp.
    # We have 2 inputs:
    #   - state_labels_DroneState.csv file  contains all the drone's state
    #   logged (acquisition_timestamp, roll, pitch, yaw, front_distance).
    #   Logging rate ~100 state/s - images/ folder: contains all the images
    #   saved with the GAP8 camera (and WiFi tx). Each image is named with its
    #   acquisition timestamp. Logging rate ~10 images/s Therefore: we normally
    #   collect the drone's state with an higher logging rate w.r.t. the image
    #   logging rate. We will end with more states than images.  This function:
    #   finds the correspondence between images and states. We only keep 1
    #   state for each image, and we discard the extra states we collected
    #   Output: a single csv file that contains (image_timestap.jpeg ,
    #   state_timestamp , roll , pitch , yaw , thrust , range.front ,
    #   mRange.rangeStatusFront)
    state_labels_files = [
        join(acquisition_path, "state_labels_DroneState.csv"),
    ]

    image_names = sorted(
        os.listdir(join(acquisition_path, "images")),
        key=lambda image_name: int(image_name.split('.')[0])
    )
    image_ticks = [int(img_name.split('.')[0]) for img_name in image_names]
    labeled_images_df = pd.DataFrame(image_names, columns=["filename"])

    for state_labels_file in state_labels_files:
        state_labels = pd.read_csv(state_labels_file)
        if "ticks" in state_labels.columns.values:
            state_labels = state_labels.rename(columns={"ticks": "timeTicks"})
            print(state_labels.columns)
        minimum_distance_idxs = []

        if(len(state_labels) == 0):
            continue

        for image_tick in image_ticks:
            minimum_distance_idxs.append(
                np.argmin(abs(state_labels["timeTicks"].values - image_tick))
            )
        close_state_samples = (
            state_labels.iloc[minimum_distance_idxs, :].reset_index(drop=True)
        )
        confName = state_labels_file.split("_")[-1].split('.')[0]
        close_state_samples = close_state_samples.rename(
            columns={"timeTicks": confName + "_TimeTicks"}
        )
        labeled_images_df = pd.concat(
            [labeled_images_df, close_state_samples],
            axis=1
        )

    return labeled_images_df

# --- WORK IN PROGRESS: a function that is meant to be used with the front distance sensor ---

# def transform_distance_to_collision_label(labeled_images_df, collision_distance=1000):
#     # Transform the distance label in a binary collision 0/1 label.
#     # Create a smaller csv file with just the information that PULP-Dronet needs:  (image.jpeg, yaw, collision_label)
#     print('I am going to label the images as collision=1 if range.front <',args.collision_distance, ', 0 otherwise')

#     # copy into a smaller pandas dataframe
#     final_labels_df = labeled_images_df[["timeTicks.jpeg", "stabilizer.yaw", "range.front"]]

#     #rename the columns with new labels
#     final_labels_df = final_labels_df.rename(columns={"stabilizer.yaw": "yaw", "range.front": "collision_label"})

#     # replace distance with a binary 0/1 label for collision:  0=no_collision, 1=collision
#     final_labels_df["collision_label"][final_labels_df["collision_label"]<collision_distance] = 1 # collision
#     final_labels_df["collision_label"][final_labels_df["collision_label"]>collision_distance] = 0 # no collision
#     return final_labels_df


if __name__ == "__main__":

    parser = create_parser()
    args = parser.parse_args()
    acquisition_folder= args.data_path
    dataset_path = os.path.join(r'../dataset/', acquisition_folder)

    # Match drone's states and images by acquisition timestamp
    labeled_images_df = label_images_gap_ticks(dataset_path)
    #save to csv file
    labeled_images_df.to_csv(os.path.join(dataset_path, "labeled_images.csv"), index=False) # no row index

    # # transform the distance label in a binary collision 0/1 label.
    # final_labels_df = transform_distance_to_collision_label(labeled_images_df, collision_distance=args.collision_distance)
    # #save to csv file
    # final_labels_df.to_csv(os.path.join(dataset_path, "final_csv_labels.csv"), index=False, header=False)  # no row index and no column header

