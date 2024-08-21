#! /usr/bin/env python
# -*- coding: utf-8 -*-
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
# File:    main.py                                                            #
# Authors:                                                                    #
#          Michal Barcis    <michal.barcis@tii.ae>                            #
#          Lorenzo Lamberti <lorenzo.lamberti@unibo.it>                       #
#          Lorenzo Bellone  <lorenzo.bellone@tii.ae>                          #
# Date:    01.03.2024                                                         #
#-----------------------------------------------------------------------------#

import argparse
import os

from dataset_visualizer.classes import Acquisition
from dataset_visualizer.video import create_video, save_all_videos
from dataset_visualizer.acquisition_visualizer import create_tk_window


def create_parser():
    parser = argparse.ArgumentParser(
        description=(
            'Image viewer for the dataset collector framework'
        )
    )
    parser.add_argument(
        '-a',
        '--acquisition',
        help=(
            'name of the aquisition to visualize; '
            'with --video you can use "all" to save all videos'
        ),
        required=True,
    )
    parser.add_argument(
        '-d',
        '--dataset_path',
        help='path to dataset folder ',
        default=os.path.join(os.path.dirname(__file__), '..', 'dataset'),
    )
    parser.add_argument(
        '--video',
        help='do not open the visualizer but create a video',
        action='store_true'
    )
    parser.add_argument(
        '--framerate',
        help='framerate for the --video option',
        default=15
    )
    parser.add_argument(
        '--mark_collision',
        action='store_true',
        help='mark all images in this acquisition as collision',
    )
    parser.add_argument(
        '--mark_no_collision',
        action='store_true',
        help='mark all images in this acquisition as no collision',
    )

    return parser


def main():
    parser = create_parser()
    args = parser.parse_args()

    if args.acquisition == "all":
        if not args.video:
            print("Only '--video' is supported for all acquisitions")
            return 1
        save_all_videos(args.dataset_path)
    else:
        acquisition_folder_path = os.path.join(
            args.dataset_path,
            args.acquisition
        )

        if not os.path.exists(acquisition_folder_path):
            print(
                "The selected aquisition does not exist. Please provide both "
                "the acquisition name and the dataset path."
            )
            return 1

        acquisition = Acquisition(
            acquisition_folder_path,
            include_deleted=True,
        )
        if not args.video:
            if args.mark_collision:
                for image in acquisition.images:
                    image.labels['collision'] = 1
                acquisition.save()

            if args.mark_no_collision:
                for image in acquisition.images:
                    image.labels['collision'] = 0
                acquisition.save()

            root = create_tk_window(acquisition)
            root.mainloop()

            acquisition.save()
        else:
            create_video(
                acquisition,
                video_name=args.acquisition
            )


if __name__ == "__main__":
    main()
