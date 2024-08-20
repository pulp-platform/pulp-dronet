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
# File:    image_visualizer.py                                                #
# Authors:                                                                    #
#          Lorenzo Lamberti <lorenzo.lamberti@unibo.it>                       #
# Date:    01.03.2024                                                         #
#-----------------------------------------------------------------------------#

import cv2

# Colors (B, G, R)
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
LIGHT_BLUE = (188, 234, 254)
DARK_BLUE = (0, 0, 254)
LIGHT_YELLOW = (105, 255, 105)
RED = (0, 0, 255)
GREEN = (0, 200, 0)
GRAY = (100, 100, 100)


def create_cv2_image(image, scale=1, highlight=False):
    yaw_rate = image.labels.get('yaw_rate')
    collision_label = image.labels.get('collision')

    # --- LOAD IMAGE ---
    img = cv2.imread(image.path, 0)
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    # get image size
    height, width = img.shape[0:2]

    yaw_rate = -int(yaw_rate)  # minus sign because axis are inverted on opencv

    # --- COLLISION ---

    # Normalize yaw and collision over image size
    margin = 10  # pixels
    collision_bar_size = height - 2*margin
    # light blue: EMPTY COLLISION BIN
    img = cv2.line(img,
                   pt1=(margin, margin),    # start_point
                   pt2=(margin, height-margin),   # end_point
                   color=LIGHT_BLUE if collision_label is not None else GRAY,
                   thickness=5)
    # dark blue: 0 = NO COLLISION, 1 COLLISION

    if collision_label is not None:
        collision_label = int(collision_label * collision_bar_size)
        img = cv2.line(
            img,
            pt1=(margin, height - margin - collision_label),  # start_point
            pt2=(margin, height-margin),   # end_point
            color=DARK_BLUE,
            thickness=5
        )

    # --- YAW RATE ---  light green
    # half circle
    height, width = img.shape[0:2]
    # Ellipse parameters.
    # NOTE: Make sure all the ellipse parameters are int otherwise it raises
    # "TypeError: ellipse() takes at most 5 arguments (8 given)"
    radius = int(40)
    center = (int(width/2), int(height - margin))
    axes = (radius, radius)
    angle = int(0)
    startAngle = int(180)
    endAngle = int(360)
    thickness = int(1)
    img = cv2.ellipse(
        img, center, axes, angle, startAngle, endAngle, WHITE, thickness
    )

    # MOVING DOT
    startAngle = int(270+yaw_rate)
    endAngle = int(270+yaw_rate+1)
    thickness = int(10)
    img = cv2.ellipse(
        img, center, axes, angle, startAngle, endAngle, LIGHT_YELLOW, thickness
    )
    if highlight:
        img = cv2.rectangle(
            img, (0, 0), (width, height), GREEN, 5, 8
        )

    if image.deleted:
        img = cv2.line(
            img,
            pt1=(0, 0),  # start_point
            pt2=(width, height),   # end_point
            color=RED,
            thickness=5
        )
        img = cv2.line(
            img,
            pt1=(width, 0),  # start_point
            pt2=(0, height),   # end_point
            color=RED,
            thickness=5
        )

    # --- RESCALE IMAGE ---
    img = cv2.resize(
        img,
        (int(width*scale), int(height*scale)),
        interpolation=cv2.INTER_AREA
    )
    return img
