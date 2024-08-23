#! /usr/bin/env python
# -*- coding: utf-8 -*-
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
# File:    plot_statistics.py                                                 #
# Authors:                                                                    #
#          Lorenzo Bellone  <lorenzo.bellone@tii.ae>                          #
#          Lorenzo Lamberti <lorenzo.lamberti@unibo.it>                       #
# Date:    01.03.2024                                                         #
#-----------------------------------------------------------------------------#

import os
from os.path import join
import argparse
import json
import pandas as pd
from classes import Dataset
import matplotlib.pyplot as plt
import numpy as np

STATS_CONFIG = join(os.path.dirname(__file__), "stats_config.json")
EXPORT_PATH = join(os.path.dirname(__file__), "plots/full_augmentation/")
if not os.path.exists(EXPORT_PATH):
    os.makedirs(EXPORT_PATH)

# Some plots restyling
plt.style.use("seaborn-dark")

def create_parser():
    parser = argparse.ArgumentParser(
        description=("Plots of statistics for the dataset")
    )

    parser.add_argument('-d', '--dataset_path',
                        help=("Dataset path with all the acquisitions."), default="../dataset")

    return parser

def cumulative_stats(df, stats):
    # Compute the cumulative stats directly from the dataframe
    stats["Scenario"]["Indoor"] = int(
        df["No_Images"][df["Scenario"] == "indoor"].sum())
    stats["Scenario"]["Outdoor"] = int(
        df["No_Images"][df["Scenario"] == "outdoor"].sum())

    stats["Path"]["Straight"] = int(
        df["No_Images"][df["Path"] == "straight"].sum())
    stats["Path"]["Turn"] = int(df["No_Images"][df["Path"] == "turns"].sum())
    stats["Path"]["Mixed"] = int(df["No_Images"][df["Path"] == "mixed"].sum())

    stats["Obstacles"]["None"] = int(
        df["No_Images"][df["Obstacles"] == "none"].sum())
    stats["Obstacles"]["Objects"] = int(
        df["No_Images"][df["Obstacles"] == "objects"].sum())
    stats["Obstacles"]["Pedestrians"] = int(
        df["No_Images"][df["Obstacles"] == "pedestrians"].sum())

    stats["Behaviour"]["Overpassing"] = int(
        df["No_Images"][df["Behaviour"] == "overpassing"].sum())
    stats["Behaviour"]["StandStill"] = int(
        df["No_Images"][df["Behaviour"] == "stand_still"].sum())
    stats["Behaviour"]["N/A"] = int(df["No_Images"]
                                    [df["Behaviour"] == "n/a"].sum())

    stats["Light"]["Bright"] = int(
        df["No_Images"][df["Light"] == "bright"].sum())
    stats["Light"]["Dark"] = int(df["No_Images"][df["Light"] == "dark"].sum())
    stats["Light"]["Normal"] = int(
        df["No_Images"][df["Light"] == "normal"].sum())
    stats["Light"]["Mixed"] = int(
        df["No_Images"][df["Light"] == "mixed"].sum())

    stats["Height"]["0.5m"] = int(df["No_Images"][df["Height"] == 0.5].sum())
    stats["Height"]["1m"] = int(df["No_Images"][df["Height"] == 1].sum())
    stats["Height"]["1.5m"] = int(df["No_Images"][df["Height"] == 1.5].sum())

    stats["No_Images"] = int(df["No_Images"].sum())

    return stats


def plot_cumulative_stats(cum_stats):
    fig, ax = plt.subplots(figsize=(12, 5))
    width = 0.2
    group_idx = 0
    images_num = cum_stats["No_Images"]
    acq_num = cum_stats["Scenario"]["Indoor"] + \
        cum_stats["Scenario"]["Outdoor"]
    del cum_stats["No_Images"]
    for outerkey, innerdict in cum_stats.items():
        bias = -width
        c = 0
        for innerkey, value in innerdict.items():
            ax.bar(group_idx + bias, value, width=width, label=innerkey)
            c += 1
            bias += width

        group_idx += 1

    ax.set_xticks(range(len(cum_stats.keys())))
    ax.set_xticklabels(cum_stats.keys())
    ax.set_yticks(range(0, images_num, 5000))
    ax.set_title("Scenarios' Statistics")
    ax.set_ylabel("Cardinality")
    ax.legend(prop={"size": 10}, bbox_to_anchor=(-0.05, 1))
    ax.grid(axis='y')

    # add info about number of images
    ax.text(5, images_num/1.1, "No Images: " + str(images_num))
    plt.tight_layout()

    plt.savefig(join(EXPORT_PATH,"Dataset_Statistics.png"), bbox_inches='tight')
    plt.show()


def combinations(df):
    comb = {"Combination": [], "Number": []}
    for q in range(2):
        for w in range(2):
            for e in range(3):
                for r in range(2):
                    for t in range(4):
                        for y in range(2):
                            comb["Combination"].append(
                                str(q) + str(w) + str(e) + str(r) + str(t) + str(y))
                            comb["Number"].append(0)

    df.loc[df["Scenario"] == "indoor", "Scenario"] = 0
    df.loc[df["Scenario"] != 0, "Scenario"] = 1

    df.loc[df["Path"] == "straight", "Path"] = 0
    df.loc[df["Path"] != 0, "Path"] = 1

    df.loc[df["Obstacles"] == "none", "Obstacles"] = 0
    df.loc[df["Obstacles"] == "objects", "Obstacles"] = 1
    df.loc[df["Obstacles"] == "pedestrians", "Obstacles"] = 2

    df.loc[df["Behaviour"] == "n/a", "Behaviour"] = 0
    df.loc[df["Behaviour"] != 0, "Behaviour"] = 1

    df.loc[df["Light"] == "bright", "Light"] = 0
    df.loc[df["Light"] == "dark", "Light"] = 1
    df.loc[df["Light"] == "normal", "Light"] = 2
    df.loc[df["Light"] == "mixed", "Light"] = 3

    df.loc[df["Height"] <= 1.0, "Height"] = 0
    df.loc[df["Height"] != 0, "Height"] = 1

    for index, acq in df.iterrows():
        acq_comb = str(acq["Scenario"]) + str(acq["Path"]) + str(acq["Obstacles"]) + \
            str(acq["Behaviour"]) + str(acq["Light"]) + str(int(acq["Height"]))
        comb["Number"][comb["Combination"].index(acq_comb)] += acq["No_Images"]

    return comb


def plot_combinations(combs):
    fig, ax = plt.subplots(figsize=(20, 8), dpi=300)
    width = 0.5

    ax.bar(combs["Combination"], combs["Number"], width=width)

    ax.set_xticks(range(len(combs["Combination"])))
    ax.set_xticklabels(combs["Combination"])
    ax.set_yticks(range(0, max(combs["Number"]) + 100, 500))
    ax.set_title("Combinations")
    ax.set_ylabel("Cardinality")

    ax.grid(axis='y')
    plt.tight_layout()

    plt.xticks(rotation=90)
    plt.savefig(join(EXPORT_PATH,"Combinations_Statistics.png"), bbox_inches='tight')
    plt.show()


def combs_to_excel(combs):

    combs["Scenario"] = []
    combs["Path"] = []
    combs["Obstacles"] = []
    combs["Behaviour"] = []
    combs["Light"] = []
    combs["Height"] = []

    for entry in combs["Combination"]:
        if entry[0] == "0":
            combs["Scenario"].append("indoor")
        else:
            combs["Scenario"].append("outdoor")

        if entry[1] == "0":
            combs["Path"].append("straight")
        else:
            combs["Path"].append("turns")

        if entry[2] == "0":
            combs["Obstacles"].append("none")
        elif entry[2] == "1":
            combs["Obstacles"].append("objects")
        else:
            combs["Obstacles"].append("pedestrians")

        if entry[3] == "0":
            combs["Behaviour"].append("n/a")
        else:
            combs["Behaviour"].append("stand_still/overpassing")

        if entry[4] == "0":
            combs["Light"].append("bright")
        elif entry[4] == "1":
            combs["Light"].append("dark")
        elif entry[4] == "2":
            combs["Light"].append("normal")
        else:
            combs["Light"].append("mixed")

        if entry[5] == "0":
            combs["Height"].append("<=1")
        else:
            combs["Height"].append(">1")

    pd.DataFrame.from_dict(combs).to_excel(join(EXPORT_PATH,"Combinations.xlsx"))


def check_labels(dataset, partition=False):
    if not partition:
        output = np.zeros((1, 5), dtype=int)
        for acquisition in dataset.acquisitions:
            for image in acquisition.images:
                if image.labels["yaw_rate"] == 0:
                    output[0][0] += 1
                if image.labels["yaw_rate"] > 0:
                    output[0][1] += 1
                if image.labels["yaw_rate"] < 0:
                    output[0][2] += 1
                if image.labels["collision"] == 1:
                    output[0][3] += 1
                if image.labels["collision"] == 0:
                    output[0][4] += 1
        print(f"""Distribution of the labels, global dataset:
                - Yaw Rate = 0: {output[0][0]}
                - Yaw Rate > 0: {output[0][1]}
                - Yaw Rate < 0: {output[0][2]}
                - Collision = 1: {output[0][3]}
                - Collision = 0: {output[0][4]}""")
    else:
        output = np.zeros((3, 5), dtype=int)
        partitions = ["train", "valid", "test"]
        for acquisition in dataset.acquisitions:
            # Load partitioned csv file
            partitioned_filename_path = join(acquisition.path, acquisition.PARTITIONED_LABELS_FILENAME)
            part_df = pd.read_csv(partitioned_filename_path)

            for partition in range(len(partitions)):
                output[partition][0] += part_df[(part_df.label_yaw_rate==0) & \
                                                (part_df.partition==partitions[partition])].count()[0]
                output[partition][1] += part_df[(part_df.label_yaw_rate>0) & \
                                                (part_df.partition==partitions[partition])].count()[0]
                output[partition][2] += part_df[(part_df.label_yaw_rate<0) & \
                                                (part_df.partition==partitions[partition])].count()[0]
                output[partition][3] += part_df[(part_df.label_collision==1) & \
                                                (part_df.partition==partitions[partition])].count()[0]
                output[partition][4] += part_df[(part_df.label_collision==0) & \
                                                (part_df.partition==partitions[partition])].count()[0]

    for j in range(output.shape[0]):
        fig, ax = plt.subplots(figsize=(15, 8), dpi=250)
        plt.grid()
        plt.ylabel("Number of Images")
        if not partition:
            plt.bar(["Yaw Rate = 0", "Yaw Rate > 0", "Yaw Rate < 0", "Collision = 1", "Collision = 0"],
                    output[0, :], width=0.5)
            plt.title("Dataset Labels distribution")
            for i, v in enumerate(output[0, :]):
                ax.text(i - .25, v + 200, str(v), color="black", fontweight="light")
            plt.savefig(join(EXPORT_PATH, "LabelsDistr.png"), bbox_inches="tight")
        else:
            plt.bar(["Yaw Rate = 0", "Yaw Rate > 0", "Yaw Rate < 0", "Collision = 1", "Collision = 0"],
                    output[j], width=0.5)
            plt.title("Labels distribution " + partitions[j] + " set")
            for i, v in enumerate(output[j]):
                ax.text(i - .25, v + 200, str(v), color="black", fontweight="light")
            plt.savefig(join(EXPORT_PATH, "LabelsDistr_" + partitions[j] + ".png"), bbox_inches="tight")



        plt.show()

def yaw_distribution(dataset):
    rates = []
    for acquisition in dataset.acquisitions:
        for image in acquisition.images:
            rates.append(image.labels["yaw_rate"])

    fig = plt.figure(figsize=(12, 6))
    plt.hist(rates, bins=90)
    plt.grid()
    plt.title("DatasetV3")
    plt.xlabel("Regression Label")
    plt.ylabel("Samples")
    plt.savefig(join(EXPORT_PATH, "Yaw_Rate_Distribution.png"), bbox_inches='tight')
    plt.show()


def main():
    parser = create_parser()
    args = parser.parse_args()

    # create a "plot/" folder if it does not already exists
    if not os.path.exists(EXPORT_PATH):
        os.makedirs(EXPORT_PATH)

    dataset = Dataset(args.dataset_path)
    dataset.initialize_from_filesystem()
    df = dataset.create_dataframe()

    yaw_distribution(dataset)

    check_labels(dataset, partition=False)
    check_labels(dataset, partition=True)

    with open(STATS_CONFIG) as f:
        stats = json.load(f)

    cum_stats = cumulative_stats(df, stats)
    with open("cumulative_stats.json", "w") as fp:
        json.dump(cum_stats, fp, indent=3)
    plot_cumulative_stats(cum_stats)

    combs = combinations(df)
    plot_combinations(combs)
    combs_to_excel(combs)


if __name__ == "__main__":
    main()
