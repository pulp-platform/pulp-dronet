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
# File:    classes.py                                                          #
# Authors:                                                                    #
#          Michal Barcis    <michal.barcis@tii.ae>                            #
#          Lorenzo Lamberti <lorenzo.lamberti@unibo.it>                       #
# Date:    01.03.2024                                                         #
#-----------------------------------------------------------------------------#

import json
import os
import pandas as pd
from os.path import join
from dataset_visualizer.dataset_post_processing import match_images_and_states

class Dataset:
    def __init__(self, path):
        self.path = path
        self.acquisitions = []

    def initialize_from_filesystem(self):
        print("Initializing dataset from the filesystem...")

        acquisition_paths = list()
        for sub_root, sub_dirs, files in os.walk(self.path):
            if 'acquisition' in  os.path.basename(sub_root):
                acquisition_paths.append(sub_root)

        for acquisition_path in acquisition_paths:
            # print(f"--- initializing acquisition '{acquisition_path}'")
            try:
                self.acquisitions.append(
                    Acquisition(acquisition_path)
                )
            except Acquisition.InitializationError:
                print(f"acquisition '{acquisition_path}' initialization error, skipping")
                continue
            # print(f"--- acquisition '{acquisition_path}' initialized successfully")
        print("Dataset successfully initialized")

    def create_dataframe(self):
        """ This method allows you to create a dataframe with all the
            characteristics of the acquisitions stored in a table. This will
            simplify the statistics' evaluation."""

        df = {"AcquisitionID": [], "Scenario": [], "Path": [], "Obstacles": [],
              "Behaviour": [], "Light": [], "Height": [], "Date": [],
              "PlaceID": [], "No_Images": []}

        if len(self.acquisitions) != 0:
            for acquisition in self.acquisitions:
                chrs = acquisition.metadata["characteristics"][0]
                df["AcquisitionID"].append(acquisition.name)
                df["Scenario"].append(chrs["scenario"])
                df["Path"].append(chrs["path"])
                df["Obstacles"].append(chrs["obstacles"])
                df["Behaviour"].append(chrs["behaviour"])
                df["Light"].append(chrs["light"])
                df["Height"].append(chrs["height"])
                df["Date"].append(chrs["date"])
                df["PlaceID"].append(chrs["place_ident"])
                df["No_Images"].append(acquisition.metadata["images_num"])

            df = pd.DataFrame(data=df)

            return df

        else:
            print("""The dataframe cannot be created since no characteristics
                     were found.\nTry to run initialize_from_filesystem() first
                     or assign characteristics to your acquisitions.""")
            return -1


class Acquisition:
    LABELS_FILENAME = "labeled_images.csv"
    PARTITIONED_LABELS_FILENAME = "labels_partitioned.csv"
    CHARACTERISTICS_FILENAME = "characteristics.json"
    LABEL_CSV_PREFIX = "label_"

    CHARACTERISTICS_FIELDS = {
        'scenario': ['indoor', 'outdoor'],
        'path': ['straight', 'turns', 'mixed'],
        'obstacles': ['none', 'pedestrians', 'objects'],
        'height': 'float',
        'behaviour': ['overpassing', 'stand_still', 'n/a'],
        'light': ['dark', 'bright', 'normal', 'mixed'],
        'place_ident': 'string',
        'date': 'date',
    }

    class InitializationError(Exception):
        pass

    def __init__(self, path, include_deleted=False):
        self.path = path
        self.images = []
        self.characteristics = [{}]

        self.images_dir_path = os.path.join(self.path, 'images')

        # probably we're loading a bit too much data while creating an object
        # it might be worth to reconsider this

        if self.has_labels():
            self.labels_from_csv(
                os.path.join(self.path, self.LABELS_FILENAME)
            )
            if include_deleted:
                self._add_images_only_in_collector_output_as_deleted()
        else:
            print(
                f"Acquisition {self.name} does not have any saved labels. "
                "Processing the cf states output."
            )
            self.initialize_images_from_collector_output()
            self.save()

        if self.has_characteristics():
            self.load_characteristics()

    @property
    def name(self):
        return os.path.basename(self.path)

    @property
    def extended_name(self):
        return os.sep.join(os.path.normpath(self.path).split(os.sep)[-3:])

    @property
    def metadata(self):
        return {
            'images_num': len(self.images),
            'characteristics': self.characteristics,
        }

    def has_labels(self):
        return os.path.exists(
            os.path.join(self.path, self.LABELS_FILENAME)
        )

    def save(self):
        self.labels_to_csv(
            os.path.join(self.path, self.LABELS_FILENAME)
        )

    @property
    def _characteristics_filename(self):
        return os.path.join(self.path, self.CHARACTERISTICS_FILENAME)

    def save_characteristics(self):
        with open(self._characteristics_filename, 'w') as f:
            json.dump(self.characteristics, f)

    def has_characteristics(self):
        return os.path.exists(
            os.path.join(self.path, self.CHARACTERISTICS_FILENAME)
        )

    def load_characteristics(self):
        with open(self._characteristics_filename) as f:
            self.characteristics = json.load(f)

    def get_images_from_collector_output(self):
        states = match_images_and_states(self.path)
        if len(states) == 0:
            print("No images in this acquisition")
            raise self.InitializationError("No images")

        state_per_image = [
            states.xs(i).to_dict()
            for i in range(len(states))
        ]
        result = []
        for state in state_per_image:
            labels = {}
            # for state_label, state_value in state.items():
            #     labels[state_label] = state_value
            try:
                labels['yaw_rate'] = state['ctrltarget.yaw']
            except KeyError:
                print("The value used for yaw_rate was not logged.")
                print("Possible options:")
                print(labels.keys())
                raise self.InitializationError("No yaw_rate")
            labels['collision'] = 0
            result.append(
                Image(
                    os.path.join(self.images_dir_path, state['filename']),
                    labels,
                    partition=None
                )
            )
        return result

    def initialize_images_from_collector_output(self):
        self.images += self.get_images_from_collector_output()

    def labels_to_csv(self, filename):
        images = [image for image in self.images if not image.deleted]
        df = pd.DataFrame({
            'filename': [image.filename for image in images],
            self.LABEL_CSV_PREFIX + 'yaw_rate': (
                [image.labels['yaw_rate'] for image in images]
            ),
            self.LABEL_CSV_PREFIX + 'collision': (
                [image.labels['collision'] for image in images]
            ),
        })
        df.to_csv(filename, index=False)

    def labels_from_csv(self, filename):
        labels = pd.read_csv(filename)
        labels_per_image = [
            labels.xs(i).to_dict()
            for i in range(len(labels))
        ]
        for labels in labels_per_image:
            self.images.append(
                Image(
                    os.path.join(self.images_dir_path, labels['filename']),
                    {
                        label[len(self.LABEL_CSV_PREFIX):]: value
                        for label, value in labels.items()
                        if label.startswith(self.LABEL_CSV_PREFIX)
                    },
                    labels['partition'] if ('partition' in labels) else None
                )
            )

    def _add_images_only_in_collector_output_as_deleted(self):
        """
        Loads images from the collector output and looks for the ones that
        are not present in self.images. Adds them to the acquisition as
        deleted images.
        """
        images = self.get_images_from_collector_output()
        for image in images:
            if image not in self.images:
                image.deleted = True
                self.images.append(image)
        self.images.sort()

class Image:
    def __init__(self, path, labels, partition):
        self.deleted = False
        self.path = path
        self.labels = labels
        self.partition = partition

    def __lt__(self, other):
        return self.name < other.name

    def __eq__(self, other):
        return self.path == other.path

    @property
    def name(self):
        return self.filename

    @property
    def filename(self):
        return os.path.basename(self.path)

    def __repr__(self):
        return f"Image <{self.path}>"
