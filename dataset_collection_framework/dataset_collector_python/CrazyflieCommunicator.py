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
# File:    CrazyflieCommunicator.py                                           #
# Authors:                                                                    #
#          Lorenzo Lamberti <lorenzo.lamberti@unibo.it>                       #
#          Daniel Rieben    <riebend@student.ethz.ch>                 #
# Date:    01.03.2024                                                         #
#-----------------------------------------------------------------------------#

import time

import cflib.crtp
import matplotlib.pyplot
import numpy as np
from cflib.crazyflie import Crazyflie
from cflib.positioning.position_hl_commander import PositionHlCommander
from cflib.crazyflie.log import LogConfig
from cflib.crazyflie.syncLogger import SyncLogger
from cflib.crazyflie.syncCrazyflie import SyncCrazyflie
from cflib.crtp.radiodriver import CRTPPacket
from multiprocessing import Process, Queue
from enum import Enum

from typing import List
from threading import Thread

from CrazyflieController import CrazyflieController, CrazylfieControlsMap, CrazyflieControllerManager


class DatasetLoggerCommand(Enum):
    START_LOGGING = 0
    STOP_LOGGING = 1
    DISCONNECT = 2
    SHUT_DOWN = 3
    CONNECT = 4
    SCAN_INTERFACES = 5
    GET_LOG_TOC = 6


class CrazyflieCommunicator:
    def __init__(self):
        self._receive_queue = Queue()
        self._command_queue = Queue()
        self._answer_queue = Queue()
        self._current_answer_id_count = 1
        self._free_answer_id_list = [0]
        self._callbacks_waiting_for_answer = dict()
        self._wait_for_answer_thread = Thread(target=self._check_answer_received, args=())

        self.communication_process = Process(target=CrazyflieCommunicator._communication_task,
                                             args=(self._command_queue,
                                                   self._receive_queue,
                                                   self._answer_queue))
        # self.communication_process.daemon = True
        self.communication_process.start()

    def send_command(self, command: DatasetLoggerCommand, cmd_data=None, answer_callback=None):
        if answer_callback is None:
            self._command_queue.put((None, command, cmd_data))
        else:
            if len(self._free_answer_id_list) > 0:
                answer_id = self._free_answer_id_list[0]
                self._free_answer_id_list = self._free_answer_id_list[1:]
            else:
                answer_id = self._current_answer_id_count
                self._current_answer_id_count += 1

            self._callbacks_waiting_for_answer[answer_id] = answer_callback
            self._command_queue.put((answer_id, command, cmd_data))
            if not self._wait_for_answer_thread.is_alive():
                self._wait_for_answer_thread = Thread(target=self._check_answer_received, args=())
                self._wait_for_answer_thread.start()

    def _check_answer_received(self):
        while len(self._callbacks_waiting_for_answer.keys()) > 0:
            answer_id, answer_data = self._get_next_queue_element(self._answer_queue, nb_elements=2)
            if answer_id is not None:
                callback = self._callbacks_waiting_for_answer.pop(answer_id)
                callback(answer_data)
                self._free_answer_id_list.append(answer_id)

    def get_packet(self) -> CRTPPacket:
        return self._receive_queue.get(block=False) if not self._receive_queue.empty() else None

    @staticmethod
    def _get_next_queue_element(queue, nb_elements=3):
        return queue.get(block=False) if not queue.empty() else ([None] * nb_elements)

    @staticmethod
    def _put_received_packet_to_queue_callback(queue: Queue, packet: CRTPPacket):
        queue.put(packet)

    @staticmethod
    def _communication_task(command_queue: Queue, receive_queue: Queue, answer_queue: Queue):
        cflib.crtp.init_drivers()
        log_configs = []
        el_id, command, cmd_data = CrazyflieCommunicator._get_next_queue_element(command_queue)
        while command != DatasetLoggerCommand.SHUT_DOWN:
            el_id, command, cmd_data = CrazyflieCommunicator._get_next_queue_element(command_queue)
            if command == DatasetLoggerCommand.CONNECT:
                with SyncCrazyflie(link_uri=cmd_data["link_uri"], cf=Crazyflie(rw_cache='./cache')) as scf:  # Open Crazyflie synchronously
                    answer_queue.put((el_id, None))
                    packet_callback = lambda packet: CrazyflieCommunicator._put_received_packet_to_queue_callback(receive_queue, packet)
                    scf.cf.packet_received.add_callback(packet_callback)
                    # =================== Controller input =========================
                    CrazyflieControllerManager.set_selected_controller_by_id(cmd_data["selected_controller_id"])
                    cf_controller = CrazyflieController(cmd_data["input_controller_map"])
                    CrazyflieControllerManager.add_queue_to_read_process(cf_controller.event_queue)
                    # =================== Controller input =========================
                    while command != DatasetLoggerCommand.DISCONNECT:
                        el_id, command, cmd_data = CrazyflieCommunicator._get_next_queue_element(command_queue)
                        if command == DatasetLoggerCommand.DISCONNECT or command == DatasetLoggerCommand.STOP_LOGGING:
                            if log_configs is not None:
                                for log_config in log_configs:
                                    log_config.stop()
                        elif command == DatasetLoggerCommand.START_LOGGING:
                            log_configs = cmd_data["log_configs"]  # type: List[LogConfig]
                            log_config_ids = []
                            for log_config in log_configs:
                                scf.cf.log.add_config(log_config)
                                log_config_ids.append((log_config.name, log_config.id))
                                log_config.start()
                            answer_queue.put((el_id, log_config_ids))
                        elif command == DatasetLoggerCommand.GET_LOG_TOC:
                            answer_queue.put((el_id, scf.cf.log.toc.toc))
                        # As long as we are connected to the crazyflie it is controlled by the controller
                        # =================== Controller input =========================
                        cf_controller.control_task(scf.cf)
                        # =================== Controller input =========================
                    # Remove callback before we disconnect
                    scf.cf.packet_received.remove_callback(packet_callback)
                    CrazyflieControllerManager.remove_queue_from_read_process(cf_controller.event_queue, False)

                        # packet = scf.cf.link.receive_packet(0)  # do not block if receive queue is empty
            elif command == DatasetLoggerCommand.SCAN_INTERFACES:
                interfaces = cflib.crtp.scan_interfaces()
                answer_queue.put((el_id, interfaces))

    def quit(self):
        self._callbacks_waiting_for_answer.clear()
        self._wait_for_answer_thread.join()
        self._command_queue.put((None, DatasetLoggerCommand.DISCONNECT, None))
        self._command_queue.put((None, DatasetLoggerCommand.SHUT_DOWN, None))
        # self.communication_process.join()
        # self.communication_process.close()


if __name__ == "__main__":
    cflib.crtp.init_drivers()

    lg_stab = LogConfig(name='Stabilizer', period_in_ms=10)
    lg_stab.add_variable('stabilizer.roll', 'float')
    lg_stab.add_variable('stabilizer.pitch', 'float')
    lg_stab.add_variable('stabilizer.yaw', 'float')
    cf_ds_logger = CrazyflieCommunicator()




