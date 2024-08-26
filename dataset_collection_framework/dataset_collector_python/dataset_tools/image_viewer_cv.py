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
# File:    image_viewer_cv.py                                                 #
# Authors:                                                                    #
#          Lorenzo Lamberti <lorenzo.lamberti@unibo.it>                       #
# Date:    01.03.2024                                                         #
#-----------------------------------------------------------------------------#

import os
from os.path import join
import sys
import glob
import json
import argparse

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from tqdm.notebook import tqdm
matplotlib.use('TkAgg')
matplotlib.use("agg")

import pandas as pd
import cv2

# Colors (B, G, R)
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
LIGHT_BLUE = (188, 234, 254)
DARK_BLUE = (0, 0, 254)
LIGHT_YELLOW=(105, 255, 105)
RED = (0, 0, 255)

def create_parser():
    parser = argparse.ArgumentParser(description='Image viewer for the dataset collector framework of the Bitcraze Crazyflie + AI-Deck')
    parser.add_argument('-f', '--folder', help='path to dataset acquisition# in the dataset folder',
                        default='acquisition13')
    parser.add_argument('-d', '--dataset_path', help='path to dataset folder ', default='../dataset/')
    parser.add_argument('--video', help='do not open the visualizer but create a video', action='store_true')
    parser.add_argument('--framerate', help='framerate for the --video option', default=15)
    return parser


def create_cv2_image(img_path,yaw_rate, collision_label):

        # --- LOAD IMAGE ---
        img = cv2.imread(img_path, 0)
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        #get image size
        height, width = img.shape[0:2]

        # --- RESCALE IMAGE ---
        # scale_image = 2
        # img = cv2.resize(img, (width*scale_image, height*scale_image), interpolation=cv2.INTER_AREA)
        # width = img.shape[1]
        # height = img.shape[0]

        # --- image margin from borders ---
        margin = 10 # pixels
        yaw_bar_size = width - 2*margin
        collision_bar_size = height - 2*margin

        # Normalize yaw and collision over image size
        collision_label = int(collision_label * collision_bar_size)
        yaw_rate = - int(yaw_rate)  #minus sign because axis are inverted on opencv
        #yaw_rate = - int(yaw_rate/10)  #minus sign because axis are inverted on opencv

        # --- COLLISION ---
        # light blue: EMPTY COLLISION BIN
        img = cv2.line(img,
                        pt1=(margin, margin),    # start_point
                        pt2=(margin, height-margin),   # end_point
                        color= LIGHT_BLUE,
                        thickness= 5)
        # dark blue: 0 = NO COLLISION, 1 COLLISION
        img = cv2.line(img,
                        pt1 =(margin, height - margin - collision_label),    # start_point
                        pt2 =(margin,height-margin),   # end_point
                        color = DARK_BLUE,
                        thickness= 5)


        # --- YAW RATE ---  light green
        # half circle
        height, width = img.shape[0:2]
        # Ellipse parameters. NOTE: Make sure all the ellipse parameters are int otherwise it raises "TypeError: ellipse() takes at most 5 arguments (8 given)"
        radius = int(40)
        center = (int(width/2), int(height - margin))
        axes = (radius, radius)
        angle = int(0)
        startAngle = int(180)
        endAngle = int(360)
        thickness = int(1)
        img = cv2.ellipse(img, center, axes, angle, startAngle, endAngle, WHITE, thickness)

        # MOVING DOT
        startAngle = int(270+yaw_rate)
        endAngle = int(270+yaw_rate+1)
        thickness = int(10)
        img = cv2.ellipse(img, center, axes, angle, startAngle, endAngle, LIGHT_YELLOW, thickness)

        # --- Instead of a moving dot i implemented a sinpler horizontal line before. ---
        # yaw_rate = - int((yaw_rate/90 * yaw_bar_size)/2)  #minus sign because axis are inverted on opencv
        # img = cv2.line(img,
        #                 pt1=(width//2, height-margin),    # start_point
        #                 pt2=(width//2 + yaw_rate, height-margin),   # end_point
        #                 color= LIGHT_YELLOW,
        #                 thickness= 5)

        return img


def plot_dataset_opencv(dataset_path):
    if not os.path.exists(join(dataset_path, 'yaw_collision.csv')):
        print('ERROR: NO LABEL FILE FOUND (name: yaw_collision.csv)')

    # --- LOAD CSV ROW ---
    labeled_imgs = pd.read_csv(join(dataset_path, 'yaw_collision.csv'))

    image_idx = 0
    number_of_imgs = len(labeled_imgs)
    while True:
        # --- GET IMAGE and IMAGE PATH---
        img_name = labeled_imgs["timeTicks.jpeg"].values[image_idx]
        img_path = join(join(dataset_path, r'images'), img_name)

        # --- GET LABELS---
        yaw_rate = labeled_imgs["yaw_rate"].values[image_idx]
        collision_label = labeled_imgs["collision_label"].values[image_idx]

        # --- Create cv2 image with overlays ---
        img = create_cv2_image(img_path,yaw_rate, collision_label)

        # --- SHOW IMAGE ---
        cv2.imshow("Preview", img)
        cv2.setWindowTitle("Preview", 'Image: %s' %(img_name)) # update title

        # --- SCROLL TO NEW IMAGE ---
        k = cv2.waitKey(100)
        if k == 27:  # Escape key
            break
        elif k == ord('d'):
            image_idx = (image_idx + 1) % number_of_imgs
        elif k == ord('a'):
            image_idx = (image_idx - 1) % number_of_imgs

    cv2.destroyAllWindows()



def create_video_opencv(dataset_path, video_name = 'video'):
    if not os.path.exists(join(dataset_path, 'yaw_collision.csv')):
        print('ERROR: NO LABEL FILE FOUND (name: yaw_collision.csv)')

    # --- LOAD CSV ROW ---
    labeled_imgs = pd.read_csv(join(dataset_path, 'yaw_collision.csv'))


    number_of_imgs = len(labeled_imgs)
    img_array = [] #this variable collects all the images that will create the video
    for image_idx in range(number_of_imgs):
        # --- GET IMAGE and IMAGE PATH---
        img_name = labeled_imgs["timeTicks.jpeg"].values[image_idx]
        img_path = join(join(dataset_path, r'images'), img_name)

        # --- GET LABELS---
        yaw_rate = labeled_imgs["yaw_rate"].values[image_idx]
        collision_label = labeled_imgs["collision_label"].values[image_idx]

        # --- Create cv2 image with overlays ---
        img = create_cv2_image(img_path,yaw_rate, collision_label)

        # --- Add image name overlay---
        img = cv2.putText(img,
                          "{}".format(img_name),
                          (150,10) ,
                          cv2.FONT_HERSHEY_SIMPLEX,
                          0.3, #font size
                          RED,
                          1)

        img_array.append(img)

    # --- CREATE VIDEO ---
    # Video name and size
    video_name = video_name + '.mp4'
    height, width = img.shape[0:2]
    size = (width,height)
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


    #### codec .avi ####
    # video_name = video_name + '.avi'
    # fourcc = cv2.VideoWriter_fourcc(*'XVID')




if __name__=="__main__":

    parser = create_parser()
    args = parser.parse_args()
    dataset_path= args.dataset_path
    acquisition_folder= args.folder

    dataset_path = 'dataset_collection_framework/dataset_collectordataset_collector_python/dataset'
    acquisition_folder_path = join(dataset_path,acquisition_folder)

    if os.path.exists(acquisition_folder_path):
        print('This acquisition folder was selected:', acquisition_folder_path)

        if not args.video:
            plot_dataset_opencv(acquisition_folder_path) # dataset vidualizer
        else:
            create_video_opencv(acquisition_folder_path, video_name=acquisition_folder ) # create a video

    else:
        print('folder not found:', acquisition_folder_path)
    print('end')

