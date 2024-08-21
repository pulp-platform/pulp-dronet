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
# File:    ds_logger.py                                                       #
# Authors:                                                                    #
#          Lorenzo Lamberti <lorenzo.lamberti@unibo.it>                       #
#          Daniel Rieben		    <riebend@student.ethz.ch>                 #
# Date:    01.03.2024                                                         #
#-----------------------------------------------------------------------------#

import logging
import time

import threading
import cflib.crtp
from cflib.crazyflie import Crazyflie
from cflib.crazyflie.log import LogConfig
from cflib.crazyflie.syncCrazyflie import SyncCrazyflie


# Only output errors from the logging framework
logging.basicConfig(level=logging.ERROR)

last_time = 0
average_time_between_packets = 0

class DatasetLogger:
    def __init__(self, cf_uri, logging_interval=50):
        # Initialize the low-level drivers (don't list the debug drivers)
        cflib.crtp.init_drivers(enable_debug_driver=False)

        self.flow_deck_attached = False
        self.uri = cf_uri
        self.log_interval = logging_interval
        self.running = False
        self.log_thread = None

        self.log_values = {
            "mRange": {
                "rFront": [],
                "rBack": [],
                "rRight": [],
                "rLeft": [],
                "rUp": [],
                "rBottom": [],
                "timestamp": []
            },
            "state": {
                "roll": [],
                "pitch": [],
                "yaw": [],
                "thrust": [],
                "timestamp": [],
                "rFront": [],
                "rBack": [],
                "rRight": [],
                "rLeft": [],
                "rUp": [],
                "rBottom": []

            }

        }

        self.log_config_state = LogConfig(name='state', period_in_ms=self.log_interval)
        self.log_config_state.add_variable('state.rfront', 'uint16_t')

        # self.log_config_range = LogConfig(name='mRange', period_in_ms=self.log_interval)
        # self.log_config_range.add_variable('mRange.rFront', 'uint16_t')
        # self.log_config_range.add_variable('mRange.rBack', 'uint16_t')
        # self.log_config_range.add_variable('mRange.rUp', 'uint16_t')
        # self.log_config_range.add_variable('mRange.rLeft', 'uint16_t')
        # self.log_config_range.add_variable('mRange.rRight', 'uint16_t')
        # self.log_config_range.add_variable('mRange.rBottom', 'uint16_t')
        # self.log_config_range.add_variable('mRange.timestamp', 'uint32_t')

    def param_deck_flow(self, name, value_str):
        value = int(value_str)
        if value:
            self.flow_deck_attached = True
            print('Flow deck is attached!')
        else:
            self.flow_deck_attached = False
            print('Flow deck is NOT attached!')

    def log_callback(self, timestamp, data, logconf):
        # for name in self.log_values[logconf.name]:
        #     self.log_values[logconf.name][name].append(data["{}.{}".format(logconf.name, name)])
        # print(self.log_values)
        a = 0.01
        global last_time
        global average_time_between_packets
        if last_time == 0:
            last_time = time.time()
        else:
            average_time_between_packets = a * (time.time() - last_time) + (1 - a) * average_time_between_packets
            last_time = time.time()
            print("Packages per second: {:.3f} pkg/s, {:.3f} ms/pkg".format(1 / average_time_between_packets,
                                                                            1000.0 * average_time_between_packets))

    def start_log_async(self, scf, logconf, log_stab_callback):
        scf.cf.log.add_config(logconf)
        logconf.data_received_cb.add_callback(log_stab_callback)
        logconf.start()

    def logging_task(self):
        with SyncCrazyflie(self.uri, cf=Crazyflie(rw_cache='./cache')) as scf:
            # Check whether flow deck is attached
            # scf.cf.param.add_update_callback(group='deck', name='bcFlow2', cb=lambda name, val_str: self.param_deck_flow(name, val_str))
            # time.sleep(1)
            # while not self.flow_deck_attached:
            #     time.sleep(5)
            #     if not self.flow_deck_attached:
            #         print("Flow deck not attached")
            self.start_log_async(scf, self.log_config_state, self.log_callback)
            # self.start_log_async(scf, self.log_config_range, self.log_callback)
            while self.running:
                pass
            self.log_config_state.stop()
            # self.log_config_range.stop()
            # Make sure that the last packet leaves before the link is closed
            # since the message queue is not flushed before closing
            scf.cf.close_link()

    def stop_logging(self):
        self.running = False

    def start_logging(self):
        self.log_thread = threading.Thread(target=lambda: self.logging_task())
        self.running = True
        self.log_thread.start()


if __name__ == '__main__':
    uri = 'radio://0/80/2M/E7E7E7E7E7'
    ds_logger = DatasetLogger(uri, logging_interval=10)
    ds_logger.start_logging()

    try:
        while True:
            pass
    except KeyboardInterrupt:
        ds_logger.stop_logging()





