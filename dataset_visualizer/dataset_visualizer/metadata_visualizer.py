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
# File:    metadata_visualizer.py                                             #
# Authors:                                                                    #
#          Michal Barcis    <michal.barcis@tii.ae>                            #
#          Lorenzo Lamberti <lorenzo.lamberti@unibo.it>                       #
#          Lorenzo Bellone  <lorenzo.bellone@tii.ae>                          #
# Date:    01.03.2024                                                         #
#-----------------------------------------------------------------------------#

import datetime
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk

from acquisition_visualizer import create_cv2_image


def display_window(acquisition):
    root = tk.Tk()
    metadata_frame = get_tk_frame(acquisition, root)
    metadata_frame.pack()

    root.mainloop()


def get_tk_frame(acquisition, root_frame):
    def retrieve():
        for entry_name, entry in entries.items():
            entry_type = acquisition.CHARACTERISTICS_FIELDS[entry_name]
            raw_value = entry.get()
            if raw_value == '':
                value = None
            elif isinstance(entry_type, list):
                value = raw_value
                assert value in entry_type
            elif entry_type == 'float':
                value = float(raw_value)
            elif entry_type == 'string':
                value = raw_value
            elif entry_type == 'date':
                try:
                    datetime.datetime.strptime(raw_value, "%Y-%m-%d")
                    value = raw_value
                except ValueError:
                    print("Incorrect date format. It should be YYYY-MM-DD")
                    continue

            else:
                raise Exception("Unknown entry type")
            acquisition.characteristics[0][entry_name] = value
        acquisition.save_characteristics()
        print("Saved")

    frame = tk.Frame(root_frame)

    entries = {
    }

    for field_name, field_value in acquisition.metadata.items():
        if field_name == 'characteristics':
            continue

        label = tk.Label(frame, text=f"{field_name}: {field_value}")
        label.pack()

    for field_name, field_value in acquisition.CHARACTERISTICS_FIELDS.items():
        label = tk.Label(frame, text=field_name)
        label.pack()

        initial_value = acquisition.characteristics[0].get(field_name, '')
        if initial_value is None:
            initial_value = ''

        if isinstance(field_value, list):
            entry = ttk.Combobox(frame, values=field_value)
            entry.set(initial_value)
        else:
            entry = tk.Entry(frame, width=20)
            entry.insert(0, initial_value)

        entry.pack()
        entries[field_name] = entry

    button = tk.Button(frame, text="save characteristics", command=retrieve)
    button.pack()
    return frame
