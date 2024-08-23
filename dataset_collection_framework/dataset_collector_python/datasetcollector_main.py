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
# File:    datasetcollector_main.py                                           #
# Authors:                                                                    #
#          Lorenzo Lamberti <lorenzo.lamberti@unibo.it>                       #
#          Daniel Rieben    <riebend@student.ethz.ch>                 #
# Date:    01.03.2024                                                         #
#-----------------------------------------------------------------------------#

import json
import os
import sys
import struct
from enum import Enum

import matplotlib.pyplot as plt
import numpy as np
from threading import Thread
from PyQt5 import QtCore
from PyQt5.Qt import QStandardItem, QStandardItemModel
from PyQt5.QtCore import Qt, pyqtSignal, QObject, QThread
from PyQt5.QtWidgets import QApplication, QMainWindow, QDialog
from PyQt5.QtGui import QImage, QPixmap, QFont, QColor
from PyQt5 import QtGui
from QLed import QLed
import pyqtgraph as pg
import io
from PIL import Image

from CrazyflieController import CrazyflieController, CrazylfieControlsMap, CrazyflieInputs, CrazyflieControllerManager
from UI.datasetcollector_ui import Ui_MainWindow
from UI.logConfigDialog import Ui_Dialog as LogConfig_Ui_Dialog
from UI.controlConfigDialog import Ui_Dialog as ControlConfig_Ui_Dialog
from typing import *


from cflib.crazyflie.log import LogConfig, Toc, LogTocElement
from CrazyflieCommunicator import CrazyflieCommunicator, DatasetLoggerCommand
from CrazyflieCameraStreamer import CrazyflieCameraStreamer


class DatasetcollectorUi(QMainWindow, Ui_MainWindow):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setupUi(self)
        self.log_config_dialog = LogConfigDialog(self)
        self.configLogButton.clicked.connect(self.on_config_button_clicked)
        self.scanButton.clicked.connect(self.on_scan_button_clicked)
        self.connectButton.clicked.connect(self.on_connect_button_clicked)
        self.connectButton.setEnabled(False)
        self.disconnectButton.clicked.connect(self.on_disconnect_button_clicked)
        self.disconnectButton.setEnabled(False)
        self.loggingButton.setCheckable(True)
        self.loggingButton.clicked.connect(self.on_log_button_clicked)
        self.loggingButton.setEnabled(False)
        self.loggingButton.setStyleSheet("QPushButton::checked{background-color : green;}")
        self.connectWifiButton.clicked.connect(self.on_wifi_connect_button_clicked)
        self.disconnectWifiButton.clicked.connect(self.on_wifi_disconnect_button_clicked)
        self.controlConfigDialog = ControlConfigDialog(self)
        self.controlConfigDialog.close_callback = self.on_close_config_dialog
        self.mapControlsButton.clicked.connect(self.on_map_controls_button_clicked)
        self.available_controllers = CrazyflieControllerManager.get_available_controllers()
        for ctrl in self.available_controllers:
            self.controllerComboBox.addItem(ctrl.get_name(), ctrl)
        selected_controller = self.controllerComboBox.itemData(self.controllerComboBox.currentIndex())
        CrazyflieControllerManager.set_selected_controller(selected_controller)
        self.cf_communicator = CrazyflieCommunicator()
        self.selected_uri = None
        self.do_logging = False
        self.logging_dir = None
        self.logging_image_dir = None
        self.log_file_name = "state_labels"
        self.log_toc = None

        self.logDataPlot.setBackground('w')
        self.logDataPlot.addLegend()
        self.plotWidgetTimer = QtCore.QTimer()

        self.logDataPlot.setWindowTitle("Log data plot")

        self.lines = dict()
        self.hist_dict = dict()
        self.all_log_confs = []
        self.log_config_file = r"./default_config.json"

        self.img_time_ticks_hist = []

        self.cf_cam_streamer = CrazyflieCameraStreamer()
        self.cam_stream_timer = QtCore.QTimer()

        self.on_scan_button_clicked()

        try:
            if os.path.isfile(r"control_config.json"):
                with open(r"control_config.json") as fp:
                    self.cf_input_map = CrazylfieControlsMap().from_json(fp)
            self.cf_control_config_file = r"control_config.json"
        except:
            self.cf_input_map = None
            self.cf_control_config_file = None

    def on_close_config_dialog(self, input_map, file_name):
        self.cf_input_map = input_map
        self.cf_control_config_file = file_name

    def on_map_controls_button_clicked(self):
        selected_controller = self.controllerComboBox.itemData(self.controllerComboBox.currentIndex())
        CrazyflieControllerManager.set_selected_controller(selected_controller)
        self.controlConfigDialog.show()

    def on_wifi_connect_button_clicked(self):
        deck_ip = self.ipAddressLineEdit.text()
        deck_port = self.portSpinBox.value()
        self.cf_cam_streamer.connect(deck_ip, deck_port)
        self.cf_cam_streamer.start_streaming()

        self.cam_stream_timer.setInterval(50)  # Update every 100 millisecond
        self.cam_stream_timer.timeout.connect(self.update_camera_image)
        self.cam_stream_timer.start()

    def on_wifi_disconnect_button_clicked(self):
        self.cam_stream_timer.stop()
        self.cf_cam_streamer.disconnect()

    def on_config_button_clicked(self):
        self.log_config_dialog.show()
        self.log_config_dialog.set_log_toc(self.log_toc)

    def update_camera_image(self):
        imgs = [pk for pk in iter(self.cf_cam_streamer.get_image_in_queue, None)]
        if len(imgs) > 0:
            img_time_ticks, imgs = [list(l) for l in zip(*[(pk[0], pk[1]) for pk in imgs])]
            # Show the last image and store the other ones
            self.img_time_ticks_hist.extend(img_time_ticks)
            if len(self.img_time_ticks_hist) > 1:
                avg_frame_rate = 1 / (np.average(np.diff(self.img_time_ticks_hist[-10:])) * 15.0e-3)
            else:
                avg_frame_rate = 0.0
            self.cameraLabel.setPixmap(DatasetcollectorUi.pil2pixmap(imgs[-1]))
            self.imageInfoLabel.setText("Framerate: {:.3f} fps - Timestamp: {}".format(avg_frame_rate, img_time_ticks[-1]))
            if self.do_logging:
                for timestmp, img in zip(img_time_ticks, imgs):
                    img.save(os.path.join(self.logging_image_dir, "{}.jpeg".format(timestmp)))

    @staticmethod
    def pil2pixmap(image):
        bytes_img = io.BytesIO()
        image.save(bytes_img, format='JPEG')
        qimg = QImage()
        qimg.loadFromData(bytes_img.getvalue())

        return QPixmap.fromImage(qimg)

    def on_disconnect_button_clicked(self):
        self.loggingButton.setEnabled(False)
        self.loggingButton.setChecked(False)
        self.connectButton.setEnabled(True)
        self.disconnectButton.setEnabled(False)
        self.scanButton.setEnabled(True)
        self.controllerComboBox.setEnabled(True)
        self.mapControlsButton.setEnabled(True)
        self.cf_communicator.send_command(DatasetLoggerCommand.STOP_LOGGING)
        self.cf_communicator.send_command(DatasetLoggerCommand.DISCONNECT)
        self.plotWidgetTimer.stop()
        self.do_logging = False

    def on_connect_button_clicked(self):
        self.connectButton.setEnabled(False)
        self.disconnectButton.setEnabled(False)
        self.scanButton.setEnabled(False)
        self.selected_uri = self.selectDeviceBox.itemText(self.selectDeviceBox.currentIndex())
        selected_controller = self.controllerComboBox.itemData(self.controllerComboBox.currentIndex())
        CrazyflieControllerManager.set_selected_controller(selected_controller)
        if self.selected_uri is not None:
            # Load default map:
            try:
                if os.path.isfile(self.cf_control_config_file):
                    with open(self.cf_control_config_file) as fp:
                        self.cf_input_map = CrazylfieControlsMap().from_json(fp)
                else:
                    self.cf_input_map = CrazylfieControlsMap(CrazyflieControllerManager.get_selected_controller())
            except:
                # Load default map
                self.cf_input_map = CrazylfieControlsMap(CrazyflieControllerManager.get_selected_controller())

            self.cf_communicator.send_command(DatasetLoggerCommand.CONNECT,
                                              cmd_data={"link_uri": self.selected_uri,
                                                        "selected_controller_id": CrazyflieControllerManager.get_selected_controller().get_id(),
                                                        "input_controller_map": self.cf_input_map},
                                              answer_callback=self.on_connected)
            self.cf_communicator.send_command(DatasetLoggerCommand.GET_LOG_TOC, answer_callback=self.on_received_log_toc)

    def on_connected(self, data):
        self.loggingButton.setEnabled(True)
        self.disconnectButton.setEnabled(True)
        self.controllerComboBox.setEnabled(False)
        self.mapControlsButton.setEnabled(False)
        print("Connected")

    def on_received_log_toc(self, toc):
        self.log_toc = toc

    def setup_log_config_from_log_groups(self, log_groups):
        self.lines = dict()
        self.all_log_confs = []
        log_group: LogGroup
        for log_group_name in log_groups.keys():
            log_group = log_groups[log_group_name]
            if log_group.log:
                log_group.group_into_configs()
                self.all_log_confs.extend(log_group.log_configs)
        nb_of_vars = np.sum([len([var for var in log_conf.variables]) for log_conf in self.all_log_confs])
        n = 0
        cmap = plt.cm.get_cmap('hsv', nb_of_vars)
        for log_conf in self.all_log_confs:
            self.lines[log_conf.name] = dict()
            for var in log_conf.variables:
                cmap_unit8 = [int(c*255.0) for c in cmap(1.0 * n / nb_of_vars)]
                pen = pg.mkPen(color=cmap_unit8)
                self.lines[log_conf.name][var.name] = self.logDataPlot.plot([], [], pen=pen, name=var.name)
                n += 1

    def on_logging_started(self, log_configs):
        for log_conf in log_configs:
            log_conf_name, assigned_id = log_conf
            conf: LogConfig
            for conf in self.all_log_confs:
                if conf.name == log_conf_name:
                    conf.id = assigned_id

    def on_log_button_clicked(self):
        if self.loggingButton.isChecked():
            if os.path.isfile(self.log_config_file):
                try:
                    self.hist_dict.clear()
                    self.lines.clear()
                    self.logDataPlot.clear()
                    self.img_time_ticks_hist = []
                    self.setup_log_config_from_log_groups(LogGroupList(self.log_config_file).log_groups)
                    self.cf_communicator.send_command(DatasetLoggerCommand.START_LOGGING, cmd_data={"log_configs": self.all_log_confs}, answer_callback=self.on_logging_started)
                    self.plotWidgetTimer.setInterval(50)  # Update every 50 millisecond
                    self.plotWidgetTimer.timeout.connect(self.update_plot_data)
                    self.plotWidgetTimer.start()
                    i = 1
                    while os.path.exists(r"./dataset/acquisition{}".format(i)):
                        i += 1

                    self.logging_dir = r"./dataset/acquisition{}".format(i)
                    self.logging_image_dir = r"./dataset/acquisition{}/images".format(i)
                    if not os.path.exists(self.logging_dir):
                        os.makedirs(self.logging_dir)
                    os.mkdir(self.logging_image_dir)
                    for log_conf in self.all_log_confs:
                        with open(os.path.join(self.logging_dir, self.log_file_name + "_" + log_conf.name + ".csv"), 'a') as file:
                            header_string = 'timeTicks,' + ','.join([var.name for var in log_conf.variables])
                            header_string += "\n"
                            file.write(header_string)
                    self.do_logging = True
                except:
                    print("Could not open file or create new dataset folders/log files")
            else:
                print("No log_config.json found. Please create a log_config.json with the created groups.")
        else:
            self.cf_communicator.send_command(DatasetLoggerCommand.STOP_LOGGING)
            self.do_logging = False

    def update_plot_data(self):
        packets = [pk for pk in iter(self.cf_communicator.get_packet, None)]
        for packet in packets:
            if packet is not None and packet.port == 5 and packet.channel == 2:
                conf_id = struct.unpack('<B', packet.data[0:1])[0]
                log_conf: LogConfig = None
                for conf in self.all_log_confs:
                    if conf.id == conf_id:
                        log_conf = conf

                if log_conf is not None:
                    time_tick = struct.unpack('<I', packet.data[1:4] + b'\x00')[0]
                    log_data = packet.data[4:]
                    unpacked_data = {}
                    data_index = 0
                    for var in log_conf.variables:
                        size = LogTocElement.get_size_from_id(var.fetch_as)
                        name = var.name
                        unpackstring = LogTocElement.get_unpack_string_from_id(var.fetch_as)
                        value = struct.unpack(unpackstring, log_data[data_index:data_index + size])[0]
                        data_index += size
                        unpacked_data[name] = value
                    if log_conf.name not in self.hist_dict.keys():
                        self.hist_dict[log_conf.name] = dict()
                        self.hist_dict[log_conf.name]["time_ticks"] = list([time_tick])
                        for key in unpacked_data.keys():
                            self.hist_dict[log_conf.name][key] = list([unpacked_data[key]])
                    else:
                        for key in unpacked_data.keys():
                            self.hist_dict[log_conf.name][key].append(unpacked_data[key])
                        self.hist_dict[log_conf.name]["time_ticks"].append(time_tick)

                    with open(os.path.join(self.logging_dir, self.log_file_name + "_" + log_conf.name + ".csv"), 'a') as file:
                        row = '{},'.format(time_tick) + ','.join([str(unpacked_data[key]) for key in unpacked_data.keys()])
                        row += '\n'
                        file.write(row)
                    for key in self.hist_dict[log_conf.name].keys():
                        if key != "time_ticks":
                            self.lines[log_conf.name][key].setData(self.hist_dict[log_conf.name]["time_ticks"], self.hist_dict[log_conf.name][key])

            else:
                if not self.do_logging:
                    self.plotWidgetTimer.start()

    def on_scan_button_clicked(self):
        self.scanButton.setEnabled(False)
        self.scanButton.setText("Scanning...")
        self.connectButton.setEnabled(False)
        self.disconnectButton.setEnabled(False)
        self.cf_communicator.send_command(DatasetLoggerCommand.SCAN_INTERFACES, answer_callback=self.on_interfaces_scanned)

    def on_interfaces_scanned(self, interfaces):
        self.selectDeviceBox.clear()
        interfaces = [interface[0] for interface in interfaces if "radio" in interface[0]]
        self.selectDeviceBox.addItems(interfaces)
        print("Found interfaces: {}".format(interfaces))
        self.connectButton.setEnabled(True)
        self.disconnectButton.setEnabled(True)
        self.scanButton.setEnabled(True)
        self.scanButton.setText("Scan")

    def stop_all_subprocesses(self):
        # CrazyflieControllerManager.quit()
        self.cf_communicator.quit()
        self.cf_cam_streamer.shutdown()

    def closeEvent(self, a0: QtGui.QCloseEvent) -> None:
        self.stop_all_subprocesses()
        a0.accept()
        self.close()

    @staticmethod
    def start_job_asynch(task, args=None, finished_action=None):
        scan_interfaces_job = Thread(target=DatasetcollectorUi.job, args=(task, args, finished_action))
        scan_interfaces_job.daemon = True
        scan_interfaces_job.start()

    @staticmethod
    def job(task, args, finished_action):
        if finished_action is not None:
            finished_action(task(args))
        else:
            task(args)

# ====================================Log Config Dialog ====================================================

class LogConfigDialog(QDialog, LogConfig_Ui_Dialog):
    byte_size = {'uint8_t': 1,
                 'uint16_t': 2,
                 'uint32_t': 4,
                 'int8_t': 1,
                 'int16_t': 2,
                 'int32_t': 4,
                 'FP16': 2,
                 'float': 4}

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setupUi(self)
        self.log_toc = None
        self.logTocTreeModel = QStandardItemModel(0, 4, parent=self.logTocTreeView)
        self.logTocTreeModel.setHorizontalHeaderLabels(["Group", "Variable", "Type", "Size"])
        self.logTocRootNode = self.logTocTreeModel.invisibleRootItem()
        self.logTocTreeView.setModel(self.logTocTreeModel)
        self.logTocTreeView.clicked.connect(self.on_log_tree_element_clicked)
        self.selected_log_tree_item = None
        self.addToGroupButton.clicked.connect(self.on_add_button_clicked)
        self.createGroupButton.clicked.connect(self.on_create_group_button_clicked)
        self.saveButton.clicked.connect(self.on_save_button_clicked)
        self.configLoadButton.clicked.connect(self.on_load_button_clicked)
        self.groups = dict()
        self.groupTreeModel = QStandardItemModel(0, 3, parent=self.groupTreeView)
        self.groupTreeModel.setHorizontalHeaderLabels(["Name", "Period [ms]", "Size"])
        self.groupTreeModel.dataChanged.connect(self.on_group_data_changed)
        self.groupRootNode = self.groupTreeModel.invisibleRootItem()
        self.groupTreeView.setModel(self.groupTreeModel)
        self.groupTreeView.clicked.connect(self.on_group_tree_element_clicked)
        self.selected_group_index = None  # type: QtCore.QModelIndex
        self.deleteButton.clicked.connect(self.on_delete_button_clicked)
        self.removeFromGroupButton.clicked.connect(self.on_remove_button_clicked)

        self.groupElementTreeModel = QStandardItemModel(0, 4, parent=self.groupElementsTreeView)
        self.groupElementTreeModel.setHorizontalHeaderLabels(["Name", "Type", "Size", "Plot Name"])
        self.groupElementTreeModel.dataChanged.connect(self.on_group_element_data_changed)
        self.groupElementRootNode = self.groupElementTreeModel.invisibleRootItem()
        self.groupElementsTreeView.setModel(self.groupElementTreeModel)

        self.on_load_button_clicked()

    def on_load_button_clicked(self):
        file_name = self.configFileNameLineEdit.text()
        with open(file_name, 'r+') as fp:
            group_config_dict = json.load(fp)
        self.groups = {k: LogGroup(k, json_string=group_config_dict[k]) for k in group_config_dict.keys()}
        for group in self.groups:
            new_item = GroupItemTreeView(item_type=GroupItemTreeViewType.GROUP_ITEM, group=self.groups[group], txt=group, set_editable=True)
            new_item.setCheckable(True)
            new_item.setCheckState(Qt.CheckState.Checked if self.groups[group].log else Qt.CheckState.Unchecked)
            ts = GroupItemTreeView(item_type=GroupItemTreeViewType.SAMPLE_RATE, group=self.groups[group], txt=str(self.groups[group].sample_period_ms), set_editable=True)
            byte_size = GroupItemTreeView(item_type=GroupItemTreeViewType.BYTE_SIZE, group=self.groups[group],  txt=str(self.groups[group].total_byte_size))
            self.groupRootNode.appendRow([new_item, ts, byte_size])

    def accept(self) -> None:
        self.on_save_button_clicked()
        self.close()

    def on_save_button_clicked(self):
        group_configs_json = json.dumps({key: self.groups[key].to_json() for key in self.groups.keys()})
        file_name = self.configFileNameLineEdit.text()
        with open(file_name, "w+") as fp:
            fp.flush()
            fp.write(group_configs_json)

    def on_group_element_data_changed(self, top_left, bottom_right, roles):
        if Qt.EditRole in roles:
            changed_data_element = self.groupElementTreeModel.itemFromIndex(top_left)
            selected_name = self.groupTreeModel.itemFromIndex(self.selected_group_index).group.name
            new_display_name = changed_data_element.data(role=Qt.DisplayRole)
            self.groups[selected_name].display_names[changed_data_element.variable.name] = new_display_name

    def on_group_data_changed(self, top_left, bottom_right, roles):
        if Qt.EditRole in roles:
            changed_data_element = self.groupTreeModel.itemFromIndex(top_left)  # type: GroupItemTreeView
            if changed_data_element.item_type == GroupItemTreeViewType.GROUP_ITEM:
                new_name = changed_data_element.data(role=Qt.DisplayRole)
                selected_name = self.groupTreeModel.itemFromIndex(self.selected_group_index).group.name
                if new_name in self.groups.keys():
                    changed_data_element.setData(selected_name, role=Qt.DisplayRole)
                else:
                    self.groups[new_name] = self.groups.pop(selected_name)
                    self.groups[new_name].name = new_name
            elif changed_data_element.item_type == GroupItemTreeViewType.SAMPLE_RATE:
                selected_name = self.groupTreeModel.itemFromIndex(self.selected_group_index).group.name
                try:
                    self.groups[selected_name].sample_period_ms = int(changed_data_element.data(role=Qt.DisplayRole))
                except:
                    pass
                changed_data_element.setData(str(self.groups[selected_name].sample_period_ms), role=Qt.DisplayRole)
        elif Qt.CheckStateRole in roles:
            changed_data_element = self.groupTreeModel.itemFromIndex(top_left)
            if changed_data_element.checkState() > 0:
                self.groups[changed_data_element.group.name].log = True
            else:
                self.groups[changed_data_element.group.name].log = False

    def on_group_tree_element_clicked(self, index: QtCore.QModelIndex):
        self.selected_group_index = index
        selected_name = self.groupTreeModel.itemFromIndex(self.selected_group_index).group.name
        self.groupElementTreeModel.clear()
        self.groupElementTreeModel = QStandardItemModel(0, 4, parent=self.groupElementsTreeView)
        self.groupElementTreeModel.setHorizontalHeaderLabels(["Name", "Type", "Size", "Plot Name"])
        self.groupElementTreeModel.dataChanged.connect(self.on_group_element_data_changed)
        self.groupElementRootNode = self.groupElementTreeModel.invisibleRootItem()
        self.groupElementsTreeView.setModel(self.groupElementTreeModel)
        for var in self.groups[selected_name].variables:
            display_name = self.groups[selected_name].display_names[var.name]
            name_el = GroupElementItemTreeView(variable=var, txt=var.name, display_name=display_name)
            type_el = GroupElementItemTreeView(variable=var, txt=var.ctype, display_name=display_name)
            byte_size = LogConfigDialog.byte_size[var.ctype]
            byte_size_el = GroupElementItemTreeView(variable=var, txt=str(byte_size), display_name=display_name)
            display_name_item = GroupElementItemTreeView(variable=var, txt=display_name, display_name=display_name, set_editable=True)
            self.groupElementRootNode.appendRow([name_el, type_el, byte_size_el, display_name_item])

    def on_add_button_clicked(self):
        if self.selected_log_tree_item is not None and self.selected_group_index is not None:
            item = self.selected_log_tree_item.item  # type: LogTocElement
            selected_group = self.groupTreeModel.itemFromIndex(self.selected_group_index).group.name
            if len(self.groups[selected_group].variables) == 0 or not any([(item.name == var.name and item.group == var.group) for var in self.groups[selected_group].variables]):
                name_el = GroupElementItemTreeView(variable=item, txt=item.name, display_name=item.name)
                type_el = GroupElementItemTreeView(variable=item, txt=item.ctype, display_name=item.name)
                byte_size = LogConfigDialog.byte_size[item.ctype]
                byte_size_el = GroupElementItemTreeView(variable=item, txt=str(byte_size), display_name=item.name)
                display_name = GroupElementItemTreeView(variable=item, txt=item.name, set_editable=True, display_name=item.name)
                self.groupElementRootNode.appendRow([name_el, type_el, byte_size_el, display_name])
                self.groups[selected_group].variables.append(item)
                self.groups[selected_group].total_byte_size += byte_size
                self.groups[selected_group].display_names[item.name] = item.name
                group_size_item = self.groupTreeModel.item(self.selected_group_index.row(), 2)
                group_size_item.setData(self.groups[selected_group].total_byte_size, role=Qt.DisplayRole)

    def on_remove_button_clicked(self):
        self.selectedGroupElementIndex = self.groupElementsTreeView.selectedIndexes()
        if len(self.selectedGroupElementIndex):
            self.selectedGroupElementIndex = self.selectedGroupElementIndex[0]
            var = self.groupElementTreeModel.itemFromIndex(self.selectedGroupElementIndex).variable
            selected_group = self.groupTreeModel.itemFromIndex(self.selected_group_index).group.name
            byte_size = LogConfigDialog.byte_size[var.ctype]
            filter_variables = lambda x: x.name != var.name
            self.groups[selected_group].variables = list(filter(filter_variables, self.groups[selected_group].variables))
            self.groupElementTreeModel.removeRow(self.selectedGroupElementIndex.row(), parent=self.selectedGroupElementIndex.parent())
            self.groups[selected_group].total_byte_size -= byte_size
            group_size_item = self.groupTreeModel.item(self.selected_group_index.row(), 2)
            group_size_item.setData(self.groups[selected_group].total_byte_size, role=Qt.DisplayRole)

    def on_create_group_button_clicked(self):
        i = 0
        new_group_name = "group{}".format(i)
        while new_group_name in self.groups.keys():
            new_group_name = "group{}".format(i)
            i += 1
        self.groups[new_group_name] = LogGroup(name=new_group_name)
        new_item = GroupItemTreeView(item_type=GroupItemTreeViewType.GROUP_ITEM, group=self.groups[new_group_name], txt=new_group_name, set_editable=True)
        new_item.setCheckable(True)
        ts = GroupItemTreeView(item_type=GroupItemTreeViewType.SAMPLE_RATE, group=self.groups[new_group_name], txt=str(self.groups[new_group_name].sample_period_ms), set_editable=True)
        byte_size = GroupItemTreeView(item_type=GroupItemTreeViewType.BYTE_SIZE, group=self.groups[new_group_name], txt=str(self.groups[new_group_name].total_byte_size))
        self.groupRootNode.appendRow([new_item, ts, byte_size])

    def on_delete_button_clicked(self):
        if self.selected_group_index is not None:
            selected_group = self.groupTreeModel.itemFromIndex(self.selected_group_index)
            self.groups.pop(selected_group.group.name)
            self.groupTreeView.clearSelection()
            self.groupTreeModel.removeRow(self.selected_group_index.row(), parent=self.selected_group_index.parent())
            selected_index = self.groupTreeView.selectedIndexes()
            self.selected_group_index = selected_index[0] if len(selected_index) > 0 else None
            self.groupElementTreeModel.clear()
            self.groupElementTreeModel = QStandardItemModel(0, 4, parent=self.groupElementsTreeView)
            self.groupElementTreeModel.setHorizontalHeaderLabels(["Name", "Type", "Size", "Plot Name"])
            self.groupElementRootNode = self.groupElementTreeModel.invisibleRootItem()
            self.groupElementsTreeView.setModel(self.groupElementTreeModel)

    def set_log_toc(self, toc: dict):
        self.log_toc = toc
        if toc is not None:
            for group in toc.keys():
                group_item = LogTocItemTreeView(txt=group)
                for var in toc[group]:
                    var_item = LogTocItemTreeView(item=toc[group][var], txt=var)
                    var_type = LogTocItemTreeView(item=None, txt=toc[group][var].ctype)
                    var_size = LogTocItemTreeView(item=None, txt=str(LogConfigDialog.byte_size[toc[group][var].ctype]))
                    group_item.appendRow([None, var_item, var_type, var_size])
                self.logTocRootNode.appendRow(group_item)

    def on_log_tree_element_clicked(self, index: QtCore.QModelIndex):
        index = self.logTocTreeView.selectedIndexes()[1]  # We always want the second column
        self.selected_log_tree_item = self.logTocTreeModel.itemFromIndex(index)
        if not isinstance(self.selected_log_tree_item, LogTocItemTreeView):
            self.selected_log_tree_item = None


class LogTocItemTreeView(QStandardItem):
    def __init__(self, item: LogTocElement = None, txt='', font_size=12, set_bold=False, color=QColor(0, 0, 0), display_name=''):
        super().__init__()
        fnt = QFont('Open Sans', font_size)
        fnt.setBold(set_bold)

        self.setEditable(False)
        self.setForeground(color)
        self.setFont(fnt)
        self.setText(txt)
        self.item = item
        self.display_name = display_name


class LogGroup:
    def __init__(self, name: AnyStr, variables: List[LogTocElement] = None, display_names=None, sample_period_ms: int=100, json_string=None):
        if json_string is not None:
            self.name = name
            self.variables = []
            self.display_names = {}
            self.sample_period_ms = sample_period_ms
            self.total_byte_size = 0
            self.log = False
            self.log_configs = []
            self.load_from_json_string(json_string)
        else:
            self.name = name
            if variables is not None:
                self.variables = variables
            else:
                self.variables = []
            if display_names is not None:
                self.display_names = display_names
            else:
                self.display_names = {}
            self.sample_period_ms = sample_period_ms
            self.total_byte_size = 0
            self.log = False
            self.log_configs = []

    def group_into_configs(self):
        if self.total_byte_size < 26:
            log_config = LogConfig(name=self.name, period_in_ms=self.sample_period_ms)
            var : LogTocElement
            for var in self.variables:
                log_config.add_variable(var.group + "." + var.name, var.ctype)
            self.log_configs.append(log_config)
        else:
            conf_size = 0
            log_conf_count = 0
            log_config = LogConfig(name=self.name + "{}".format(log_conf_count), period_in_ms=self.sample_period_ms)
            var: LogTocElement
            for var in self.variables:
                conf_size += LogConfigDialog.byte_size[var.ctype]
                if conf_size <= 26:  # Maximum size allowed by bitcraze firmware
                    log_config.add_variable(var.group + "." + var.name, var.ctype)
                else:
                    self.log_configs.append(log_config)
                    log_conf_count += 1
                    log_config = LogConfig(name=self.name + "{}".format(log_conf_count), period_in_ms=self.sample_period_ms)
                    log_config.add_variable(var.group + "." + var.name, var.ctype)
                    conf_size = LogConfigDialog.byte_size[var.ctype]
            self.log_configs.append(log_config)

    def to_json(self):
        var_dict = {}
        for var in self.variables:
            var_dict[var.group + "." + var.name] = {
                "ident": var.ident,
                "ctype": var.ctype,
                "pytype": var.pytype,
                "access": var.access
            }
        log_group_dict = {"sample_period_ms": self.sample_period_ms,
                          "total_byte_size": self.total_byte_size,
                          "variables": var_dict,
                          "display_names": self.display_names,
                          "log": self.log
                          }
        return json.dumps(log_group_dict)

    def load_from_json_string(self, log_group_json):
        log_group_dict = json.loads(log_group_json)
        self.sample_period_ms = log_group_dict["sample_period_ms"]
        self.total_byte_size = log_group_dict["total_byte_size"]
        self.display_names = log_group_dict["display_names"]
        self.log = log_group_dict["log"]
        self.variables = []
        for var in log_group_dict["variables"].keys():
            toc_el = LogTocElement()
            group, name = var.split(".")
            toc_el.group = group
            toc_el.name = name
            toc_el.ident = log_group_dict["variables"][var]["ident"]
            toc_el.ctype = log_group_dict["variables"][var]["ctype"]
            toc_el.pytype = log_group_dict["variables"][var]["pytype"]
            toc_el.access = log_group_dict["variables"][var]["access"]
            self.variables.append(toc_el)


class LogGroupList:
    def __init__(self, file_path: str = None):
        if file_path is not None:
            self.log_groups = dict()
            self.load_from_file(file_path)
        else:
            self.log_groups = dict()

    def load_from_file(self, file_path: str):
        with open(file_path, 'r') as fp:
            self.log_groups = json.load(fp)
        self.log_groups = {key: LogGroup(key, json_string=self.log_groups[key]) for key in self.log_groups.keys()}


class GroupItemTreeViewType(Enum):
    GROUP_ITEM = 0
    SAMPLE_RATE = 1
    BYTE_SIZE = 2


class GroupItemTreeView(QStandardItem):
    def __init__(self, item_type: GroupItemTreeViewType, group: LogGroup = None, txt='', font_size=12, set_bold=False, color=QColor(0, 0, 0), set_editable=False):
        super().__init__()
        fnt = QFont('Open Sans', font_size)
        fnt.setBold(set_bold)

        self.setEditable(set_editable)
        self.setForeground(color)
        self.setFont(fnt)
        self.setText(txt)
        self.group = group
        self.item_type = item_type


class GroupElementItemTreeView(QStandardItem):
    def __init__(self, variable: LogTocElement, txt='', font_size=12, set_bold=False, color=QColor(0, 0, 0), set_editable=False, display_name=''):
        super().__init__()
        fnt = QFont('Open Sans', font_size)
        fnt.setBold(set_bold)

        self.setEditable(set_editable)
        self.setForeground(color)
        self.setFont(fnt)
        self.setText(txt)
        self.variable = variable
        self.display_name = display_name

# ====================================Log Config Dialog End ====================================================

# ====================================Control Config Dialog ====================================================

class ControlConfigDialog(QDialog, ControlConfig_Ui_Dialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setupUi(self)
        self.input_map = CrazylfieControlsMap(CrazyflieControllerManager.get_selected_controller())
        self.cf_controller = CrazyflieController(self.input_map)
        self.assign_mode = False
        self.assistButtonLed.setOnColour(QLed.Green)
        self.assign_control = CrazyflieInputs.NOT_ASSIGNED
        self.assignYawButton.clicked.connect(self.on_assign_yaw_rate)
        self.assignPitchButton.clicked.connect(self.on_assign_vx)
        self.assignRollButton.clicked.connect(self.on_assign_vy)
        self.assignVzButton.clicked.connect(self.on_assign_vz)
        self.assignAssistButton.clicked.connect(self.on_assign_assist_button)
        self.input_events = []
        self.update_gui_thread = None
        self.update_gui_worker = None
        self.close_callback = None
        self.config_file_path = "control_config.json"

    def showEvent(self, a0: QtGui.QShowEvent) -> None:
        cf_coord_image = Image.open(r"./UI/coordinate_system.jpg").resize((self.crazyflieCoordImage.width(), self.crazyflieCoordImage.height()))
        self.crazyflieCoordImage.setPixmap(DatasetcollectorUi.pil2pixmap(cf_coord_image))

        self.config_file_path = self.controlConfigFileNameLineEdit.text()
        if os.path.isfile(self.config_file_path):
            try:
                with open(self.config_file_path) as fp:
                    self.input_map.from_json(fp)
            except:
                self.input_map = CrazylfieControlsMap(CrazyflieControllerManager.get_selected_controller())
                print("Couldn't read config file! Loading Default.")
        self.cf_controller = CrazyflieController(self.input_map)
        CrazyflieControllerManager.register_callback(self.on_joystick_input)
        # print("Thread running: {}".format(self.update_gui_thread.isRunning()))
        self.update_gui_thread = QThread()
        self.update_gui_worker = UpdateControllerMapGuiWorker()
        self.update_gui_worker.run = True
        self.update_gui_worker.moveToThread(self.update_gui_thread)
        self.update_gui_worker.value_changed_signal.connect(self.update_control_gui)
        self.update_gui_thread.started.connect(self.update_gui_worker.update_gui_task)
        self.update_gui_thread.start()
        # self.controlConfigFileNameLineEdit

    def closeEvent(self, a0: QtGui.QCloseEvent) -> None:
        CrazyflieControllerManager.unregister_callback(self.on_joystick_input)
        self.update_gui_worker.run = False
        self.update_gui_thread.quit()
        self.update_gui_thread.wait()

    def accept(self) -> None:
        with open(self.config_file_path, "w") as fp:
            fp.write(self.cf_controller.input_map.to_json())
        if self.close_callback is not None:
            self.close_callback(self.cf_controller.input_map, self.config_file_path)
        self.close()

    def reject(self) -> None:
        self.close()

    def on_assign_vx(self):
        self.assign_control = CrazyflieInputs.PITCH
        self.cf_controller.unassign_crazyflie_input(self.assign_control)
        self.assign_mode = True

    def on_assign_vy(self):
        self.assign_control = CrazyflieInputs.ROLL
        self.cf_controller.unassign_crazyflie_input(self.assign_control)
        self.assign_mode = True

    def on_assign_vz(self):
        self.assign_control = CrazyflieInputs.THRUST
        self.cf_controller.unassign_crazyflie_input(self.assign_control)
        self.assign_mode = True

    def on_assign_yaw_rate(self):
        self.assign_control = CrazyflieInputs.YAW
        self.cf_controller.unassign_crazyflie_input(self.assign_control)
        self.assign_mode = True

    def on_assign_assist_button(self):
        self.assign_control = CrazyflieInputs.ASSISTED_MODE
        self.cf_controller.unassign_crazyflie_input(self.assign_control)
        self.assign_mode = True

    def on_joystick_input(self, event):
        if self.assign_mode:
            if self.cf_controller.assign_map(event, self.assign_control):
                self.assign_mode = False
        else:
            e_mapped = self.cf_controller.map_input(event)
            if e_mapped is not None:
                self.input_events.append(e_mapped)
                self.update_gui_worker.value_changed = True

    def clamp_slider_value(self, value):
        return max(min(value, 100), -100)

    def update_control_gui(self):
        if len(self.input_events) > 0:
            for e in self.input_events:
                if e.input_type == CrazyflieInputs.ROLL:
                    # print("Roll: {}".format(e_mapped.value))
                    self.rollSlider.setValue(self.clamp_slider_value(int(e.value * 100)))
                elif e.input_type == CrazyflieInputs.PITCH:
                    # print("Pitch: {}".format(e_mapped.value))
                    self.pitchSlider.setValue(self.clamp_slider_value(int(e.value * 100)))
                elif e.input_type == CrazyflieInputs.YAW:
                    # print("Yaw: {}".format(e_mapped.value))
                    self.yawSlider.setValue(self.clamp_slider_value(int(e.value * 100)))
                elif e.input_type == CrazyflieInputs.THRUST:
                    # print("Thrust: {}".format(e_mapped.value))
                    self.upDownSlider.setValue(self.clamp_slider_value(int(e.value * 100)))
                elif e.input_type == CrazyflieInputs.ASSISTED_MODE:
                    self.assistButtonLed.setValue(int(e.value))

            self.input_events = []


class UpdateControllerMapGuiWorker(QObject):
    value_changed_signal = pyqtSignal()
    def __init__(self):
        super().__init__()
        self.value_changed = False
        self.run = True

    def update_gui_task(self):
        while self.run:
            if self.value_changed:
                self.value_changed = False
                self.value_changed_signal.emit()


# ====================================Control Config Dialog End ====================================================

if __name__ == "__main__":

    app = QApplication(sys.argv)
    win = DatasetcollectorUi()
    win.show()
    try:
        sys.exit(app.exec())
    except KeyboardInterrupt:
        DatasetcollectorUi.stop_all_subprocesses()
        pass

