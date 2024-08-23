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
# File:    imageViewerUI.py                                                   #
# Authors:                                                                    #
#          Lorenzo Lamberti <lorenzo.lamberti@unibo.it>                       #
# Date:    01.03.2024                                                         #
#-----------------------------------------------------------------------------#

import sys
import glob
import json
import argparse

import pandas as pd
import os
from dataset_post_processing import label_images_gap_ticks
# from dataset_post_processing import label_images
import matplotlib.pyplot as plt
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QApplication, QMainWindow, QFileDialog
from PyQt5.QtGui import QImage, QPixmap
from PyQt5 import QtGui
import pyqtgraph as pg
import io
from PIL import Image, ImageDraw

from UI.imageViewerUI import Ui_MainWindow


# import cv2
# for k, v in os.environ.items():
#     if k.startswith("QT_") and "cv2" in v:
#         del os.environ[k]

# matplotlib.use('TkAgg')

def create_parser():
    parser = argparse.ArgumentParser(description='Image viewer for the dataset collector framework of the Bitcraze Crazyflie + AI-Deck')
    parser.add_argument('-d', '--data_path', help='path to dataset acquisition# in the ../dataset/ folder',
                        default='acquisition2')
    return parser

class ImageViewer(QMainWindow, Ui_MainWindow):
    def __init__(self, dataset_path=None, parent=None):
        super().__init__(parent)
        self.setupUi(self)
        self.keyPressEvent = self.on_key_press_event
        self.state_labels_files = None
        self.state_labels_data = []
        self.labeled_image_data = None
        self.lines = dict()
        self.dataPlot.setBackground('w')
        self.dataPlot.addLegend()
        self.dataset_dir = dataset_path
        self.load_data()

        # if not os.path.isfile(r"image_viewer_state.json"):
        #     with open(r"image_viewer_state.json", "w") as fp:
        #         fp.write(json.dumps({"dataset_dir": None}))
        #     self.dataset_dir = None
        # else:
        #     with open(r"image_viewer_state.json", "r") as fp:
        #         self.dataset_dir = json.load(fp)["dataset_dir"]
        #     self.load_data()
        self.loadDatasetButton.clicked.connect(self.on_load_button_clicked)
        self.current_image_index = 0

    def on_key_press_event(self, e: QtGui.QKeyEvent):
        n_max = len(self.labeled_image_data)
        if n_max > 0:
            if e is not None:
                if e.key() == Qt.Key_D:
                    self.current_image_index = (self.current_image_index + 1) % n_max
                elif e.key() == Qt.Key_A:
                    self.current_image_index = (self.current_image_index - 1) % n_max
            labeled_img_row = list(self.labeled_image_data.iloc[self.current_image_index, :].values)
            col_names = list(self.labeled_image_data.columns.values)
            info_string = ''.join([(col_name + "=" + str(data) + "\n") for (col_name, data) in zip(col_names, labeled_img_row)])
            self.textEdit.setText(info_string)
            img_name = labeled_img_row[0]
            img_time_tick = int(img_name.split('.')[0])
            img = Image.open(os.path.join(self.dataset_dir, "images", img_name))
            if "range.front" in col_names or "range_front[mm]" in col_names:
                if "range.front" in col_names:
                    range_front = self.labeled_image_data["range.front"].iloc[self.current_image_index]
                else:
                    range_front = self.labeled_image_data["range_front[mm]"].iloc[self.current_image_index]
                imgrgba = Image.new("RGB", img.size)
                imgrgba.paste(img)
                img = imgrgba
                draw = ImageDraw.Draw(img)
                collision_val = int(min(255.0 * 300 / (range_front if range_front != 0 else 1), 255))
                draw.ellipse((10, 10, 40, 40), fill=(collision_val, 0, 0), outline=(collision_val, 0, 0))
            self.imageLabel.setPixmap(ImageViewer.pil2pixmap(img))
            state_labels: pd.DataFrame
            for state_labels in self.state_labels_data:
                cropped_data = state_labels.where(state_labels.iloc[:, 0] <= img_time_tick).dropna()
                for var_name in cropped_data.columns.values[1:]:
                    self.lines[var_name].setData(cropped_data.iloc[:, 0], cropped_data[var_name])

    def on_load_button_clicked(self):
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        self.dataset_dir = QFileDialog.getExistingDirectory(self, "Choose dataset directory...", "../dataset", options=options)
        self.setWindowTitle(self.dataset_dir.split('/')[-1])
        self.load_data()

    def load_data(self):
        if self.dataset_dir is not None and os.path.isdir(self.dataset_dir):
            self.state_labels_data = []
            self.lines.clear()
            self.dataPlot.clear()
            self.current_image_index = 0
            self.state_labels_files = glob.glob(os.path.join(self.dataset_dir, "state_labels*.csv"))
            for file in self.state_labels_files:
                self.state_labels_data.append(pd.read_csv(file))
            nb_of_vars = 0
            for state_labels in self.state_labels_data:
                nb_of_vars += len(state_labels.columns.values) - 1  # Each file has one time column
            n = 0
            cmap = plt.cm.get_cmap('hsv', nb_of_vars)
            for state_labels in self.state_labels_data:
                for var_name in state_labels.columns.values[1:]:  # First column are always the time ticks
                    cmap_unit8 = [int(c * 255.0) for c in cmap(1.0 * n / nb_of_vars)]
                    pen = pg.mkPen(color=cmap_unit8)
                    self.lines[var_name] = self.dataPlot.plot([], [], pen=pen, name=var_name)
                    n += 1
            if not os.path.isfile(os.path.join(self.dataset_dir, "labeled_images.csv")):
                label_images_gap_ticks(self.dataset_dir)
            self.labeled_image_data = pd.read_csv(os.path.join(self.dataset_dir, "labeled_images.csv"))
            self.on_key_press_event(None)
        else:
            self.setWindowTitle("No Dataset Loaded")
            print("Please choose first a dataset directory!")

    @staticmethod
    def pil2pixmap(image):
        bytes_img = io.BytesIO()
        image.save(bytes_img, format='png')
        qimg = QImage()
        qimg.loadFromData(bytes_img.getvalue())

        return QPixmap.fromImage(qimg)

    def closeEvent(self, a0: QtGui.QCloseEvent) -> None:
        with open(r"image_viewer_state.json", "w") as fp:
            fp.flush()
            fp.write(json.dumps({"dataset_dir": self.dataset_dir}))
        a0.accept()



if __name__=="__main__":
    parser = create_parser()
    args = parser.parse_args()
    acquisition_folder= args.data_path
    dataset_path = os.path.join(r'../dataset/', acquisition_folder)

    app = QApplication(sys.argv)
    win = ImageViewer(dataset_path=dataset_path)
    win.show()
    try:
        sys.exit(app.exec())
    except KeyboardInterrupt:
        pass


