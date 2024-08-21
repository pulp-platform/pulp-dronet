#! /usr/bin/env python
# -*-coding: utf-8 -*-
# #-----------------------------------------------------------------------------#
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
# File:    acquisition_visualizer.py                                          #
# Authors:                                                                    #
#          Michal Barcis    <michal.barcis@tii.ae>                            #
#          Lorenzo Lamberti <lorenzo.lamberti@unibo.it>                       #
#          Lorenzo Bellone  <lorenzo.bellone@tii.ae>                          #
# Date:    01.03.2024                                                         #
#-----------------------------------------------------------------------------#

import cv2
import numpy as np
import tkinter as tk
from PIL import Image, ImageTk

from image_visualizer import create_cv2_image
import metadata_visualizer


class AcquisitionVisualizer:
    def __init__(self, acquisition, root_frame):
        self.acquisition = acquisition
        self.image_ident = 0
        self.setup_frame(root_frame)
        self.bind_keys(root_frame)
        self.animating = False
        self.showing_grid = False

        self.toggle_animation()

        print("*"*10)
        print("Use 'a' and 'd' to switch to previous/next image.")
        print("Use 'w' to mark collision.")
        print("Use 's' to mark no collision.")
        print("Use 'i' to delete an erroneous yaw rate")
        print("Use 'l' to go left")
        print("Use 'r' to go right")
        print("Use 'q' to delete the image from the dataset "
              "(it won't be displayed next time the program is started)")
        print("Use 'e' to un-delete the image")
        print("Everything is saved as soon as you change the value.")
        print("Have fun!")
        print("*"*10)

    def setup_frame(self, root_frame):
        self.frame = tk.Frame(root_frame)
        self._label = tk.Label(self.frame)
        self._label.pack()
        self.setup_buttons()

    def toggle_grid_view(self):
        self.showing_grid = not self.showing_grid
        if not self.showing_grid:
            cv2.destroyAllWindows()
        self.refresh()

    def show_grid(self):
        grid = create_image_grid(self.acquisition, self.image_ident)
        cv2.namedWindow("GridView", cv2.WINDOW_NORMAL)
        cv2.imshow("GridView", grid)
        cv2.setWindowTitle("GridView", 'GridView')  # update title
        k = cv2.waitKey(100)

    def setup_buttons(self):
        buttons = {
            'toggle animation (p)': self.toggle_animation,
            'GridView': self.toggle_grid_view,
        }

        for text, action in buttons.items():
            button = tk.Button(self.frame, text=text, command=action)
            button.pack()

    def bind_keys(self, root):
        root.bind("<a>", lambda _: self.previous_image())
        root.bind("<d>", lambda _: self.next_image())
        root.bind("<w>", lambda _: self.mark_collision(1))
        root.bind("<s>", lambda _: self.mark_collision(0))
        root.bind("<e>", lambda _: self.mark_deleted(False))
        root.bind("<q>", lambda _: self.mark_deleted(True))
        root.bind("<p>", lambda _: self.toggle_animation())
        root.bind("<i>", lambda _: self.mark_straight())
        root.bind("<l>", lambda _: self.mark_turn("left"))
        root.bind("<r>", lambda _: self.mark_turn("right"))

    def mark_collision(self, value):
        assert value in [0, 1]
        self._image.labels['collision'] = value
        self.refresh()
        self.acquisition.save()

    def mark_straight(self):
        self._image.labels['yaw_rate'] = 0.0
        self.refresh()
        self.acquisition.save()

    def mark_turn(self, direction):
        assert direction in ["left", "right"]
        if direction == "left":
            self._image.labels["yaw_rate"] = 90
        else:
            self._image.labels["yaw_rate"] = -90

        self.refresh()
        self.acquisition.save()

    def mark_deleted(self, value):
        assert value in [True, False]
        self._image.deleted = value
        self.refresh()
        self.acquisition.save()

    def toggle_animation(self):
        if self.animating:
            self.animating = 'stop'
        else:
            self.animating = True

            def loop():
                self.next_image()
                if self.animating == 'stop':
                    self.animating = False
                elif self.animating:
                    self._label.after(200, loop)
            loop()

    def next_image(self):
        self.image_ident = (
            (self.image_ident + 1) % len(self.acquisition.images)
        )
        self.refresh()

    def previous_image(self):
        self.image_ident = (
            (self.image_ident - 1) % len(self.acquisition.images)
        )
        self.refresh()

    @property
    def _image(self):
        return self.acquisition.images[self.image_ident]

    def refresh(self):
        img = create_cv2_image(self._image)
        blue,green,red = cv2.split(img)
        img = cv2.merge((red, green, blue))
        pil_img = Image.fromarray(img)
        imgtk = ImageTk.PhotoImage(image=pil_img)
        self._label.imgtk = imgtk
        self._label.configure(image=imgtk)
        if self.showing_grid:
            self.show_grid()


def create_image_grid(acquisition, highlight_idx=None):
    images = list(enumerate(acquisition.images))

    n = len(images)
    w = int(np.ceil(np.sqrt(n)))
    h = int(np.ceil(n/w))

    rows_imgs = [
        images[i*w:(i+1)*w]
        for i in range(h)
    ]

    rows = [
        np.hstack(tuple(
            create_cv2_image(image, scale=1, highlight=(idx == highlight_idx))
            for idx, image in row
        ))
        for row in rows_imgs
    ]
    last_row = rows[-1]
    padded_last_row = np.zeros_like(rows[0])
    padded_last_row[:last_row.shape[0], :last_row.shape[1]] = last_row
    rows[-1] = padded_last_row

    return np.vstack(tuple(rows))


def create_tk_window(acquisition, root=None):
    if root is None:
        root = tk.Tk()
    visualizer = AcquisitionVisualizer(acquisition, root)
    visualizer.frame.pack(side=tk.LEFT)

    metadata_frame = metadata_visualizer.get_tk_frame(acquisition, root)
    metadata_frame.pack(side=tk.LEFT)

    return root

    root.mainloop()
