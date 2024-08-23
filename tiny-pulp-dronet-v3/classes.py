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
# File:    classes.py                                                         #
# Authors:                                                                    #
#          Michal Barcis    <michal.barcis@tii.ae>                            #
#          Lorenzo Lamberti <lorenzo.lamberti@unibo.it>                       #
#          Lorenzo Bellone  <lorenzo.bellone@tii.ae>                          #
# Date:    01.03.2024                                                         #
#-----------------------------------------------------------------------------#

import json
import os
from os.path import join
import pandas as pd


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

        self.images_dir_path = join(self.path, 'images')

        # probably we're loading a bit too much data while creating an object
        # it might be worth to reconsider this

        if self.has_labels():
            self.labels_from_csv(
                join(self.path, self.LABELS_FILENAME)
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
            join(self.path, self.LABELS_FILENAME)
        )

    def save(self):
        self.labels_to_csv(
            join(self.path, self.LABELS_FILENAME)
        )

    @property
    def _characteristics_filename(self):
        return join(self.path, self.CHARACTERISTICS_FILENAME)

    def save_characteristics(self):
        with open(self._characteristics_filename, 'w') as f:
            json.dump(self.characteristics, f)

    def has_characteristics(self):
        return os.path.exists(
            join(self.path, self.CHARACTERISTICS_FILENAME)
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
                    join(self.images_dir_path, state['filename']),
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
                    join(self.images_dir_path, labels['filename']),
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
