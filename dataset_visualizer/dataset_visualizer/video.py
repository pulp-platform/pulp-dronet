#! /usr/bin/env python
# -*- coding: utf-8 -*-
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
# File:    video.py                                                           #
# Authors:                                                                    #
#          Lorenzo Lamberti <lorenzo.lamberti@unibo.it>                       #
#          Michal Barcis    <michal.barcis@tii.ae>                            #
#          Lorenzo Bellone  <lorenzo.bellone@tii.ae>                          #
# Date:    01.03.2024                                                         #
#-----------------------------------------------------------------------------#

import os
from os.path import join
import cv2

from dataset_visualizer.classes import Dataset
from dataset_visualizer.image_visualizer import create_cv2_image, RED

def save_all_videos(dataset_path):
    d = Dataset(dataset_path)
    d.initialize_from_filesystem()
    for acquisition in d.acquisitions:
        print(f"Saving video for acquisition '{acquisition.name}'")
        create_video(acquisition, acquisition.name)
    print("All videos saved")

def create_video(acquisition, video_name='video'):
    number_of_imgs = len(acquisition.images)
    img_array = []  # this variable collects all the images that will create the video
    for image_idx in range(number_of_imgs):
        # --- GET IMAGE and IMAGE PATH---
        image = acquisition.images[image_idx]
        img_name = image.name

        # --- Create cv2 image with overlays ---
        img = create_cv2_image(image)

        # --- Add image name overlay---
        img = cv2.putText(img,
                          "{}".format(img_name),
                          (150, 10),
                          cv2.FONT_HERSHEY_SIMPLEX,
                          0.3,  # font size
                          RED,
                          1)

        img_array.append(img)

    # --- CREATE VIDEO ---
    # Video name and size
    video_name = join(os.path.dirname(acquisition.path) , video_name+'.mp4')
    height, width = img.shape[0:2]
    size = (width, height)
    # Define the codec
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    # Create VideoWriter object
    video_out = cv2.VideoWriter(video_name, fourcc, 15, size)
    # Write video
    print('processing the video....')
    for i in range(len(img_array)):
        video_out.write(img_array[i])
    video_out.release()
    print('video saved!')
