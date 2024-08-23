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
# File:    CrazyflieController.py                                             #
# Authors:                                                                    #
#          Lorenzo Lamberti <lorenzo.lamberti@unibo.it>                       #
#          Daniel Rieben    <riebend@student.ethz.ch>                 #
# Date:    01.03.2024                                                         #
#-----------------------------------------------------------------------------#

import time
from multiprocessing import Process, Queue
import pygame
from pygame.joystick import Joystick
from typing import List
from enum import Enum
import json
from threading import Thread
import numpy as np


class JoyInputType(Enum):
    BUTTON = "button"
    AXIS = "axis"
    BALL = "ball"
    HAT = "hat"


class CrazyflieInputs(Enum):
    NOT_ASSIGNED = "None"
    YAW = "yaw"
    PITCH = "pitch"
    ROLL = "roll"
    THRUST = "thrust"
    ASSISTED_MODE = "assist_mode"

class ControllerInputEvent:
    def __init__(self, event):
        for key in event.__dict__.keys():
            setattr(self, key, getattr(event, key))
        if hasattr(event, "type"):
            self.type = event.type


class CrazylfieControlsMap(dict):
    def __init__(self, joystick: Joystick = None):
        super().__init__()
        if joystick is not None:
            if "Logitech" in joystick.get_name():
                self[JoyInputType.AXIS] = dict()
                # Set some default settings
                self[JoyInputType.AXIS][0] = (CrazyflieInputs.YAW, 1)
                self[JoyInputType.AXIS][1] = (CrazyflieInputs.THRUST, 1)
                self[JoyInputType.AXIS][2] = (CrazyflieInputs.ROLL, 1)
                self[JoyInputType.AXIS][3] = (CrazyflieInputs.PITCH, 1)
                self[JoyInputType.BUTTON] = dict()
                # Set default assist mode button
                self[JoyInputType.BUTTON][5] = (CrazyflieInputs.ASSISTED_MODE, 1)
                self[JoyInputType.HAT] = dict()
                self[JoyInputType.BALL] = dict()
            else:  # Take the Sony mapping as default
                print("Load default Sony mapping")
                self[JoyInputType.AXIS] = dict()  # [CrazyflieInputs.NOT_ASSIGNED] * joystick.get_numaxes()
                # Set some default settings
                self[JoyInputType.AXIS][0] = (CrazyflieInputs.YAW, 1)
                self[JoyInputType.AXIS][1] = (CrazyflieInputs.THRUST, 1)
                self[JoyInputType.AXIS][3] = (CrazyflieInputs.ROLL, 1)
                self[JoyInputType.AXIS][4] = (CrazyflieInputs.PITCH, 1)
                self[JoyInputType.BUTTON] = dict()  # [CrazyflieInputs.NOT_ASSIGNED] * joystick.get_numbuttons()
                # Set default assist mode button
                self[JoyInputType.BUTTON][5] = (CrazyflieInputs.ASSISTED_MODE, 1)
                self[JoyInputType.HAT] = dict()  # [CrazyflieInputs.NOT_ASSIGNED] * joystick.get_numhats()
                self[JoyInputType.BALL] = dict()  # [CrazyflieInputs.NOT_ASSIGNED] * joystick.get_numballs()
        else:  # Initialize wih some values
            self[JoyInputType.AXIS] = dict()
            self[JoyInputType.BUTTON] = dict()
            self[JoyInputType.HAT] = dict()
            self[JoyInputType.BALL] = dict()

    def set_input(self, input_type: JoyInputType, input_id: int, cf_input_type: CrazyflieInputs):
        self[input_type][input_id] = cf_input_type

    def from_json(self, json_file):
        json_dict = json.load(json_file)
        self.load_from_dict(json_dict)
        return self

    def from_json_string(self, json_string):
        json_dict = json.loads(json_string)
        self.load_from_dict(json_dict)
        return self

    def load_from_dict(self, controller_map: dict):
        self.clear()
        for key in controller_map.keys():
            self[JoyInputType(key)] = dict()
            for control_id in controller_map[key].keys():
                input_type = CrazyflieInputs(controller_map[key][control_id][0])
                sign = controller_map[key][control_id][1]
                self[JoyInputType(key)][int(control_id)] = (input_type, sign)
        return self

    def to_json(self):
        json_dict_compatiple = dict()
        for key in self.keys():
            json_dict_compatiple[key.value] = dict()
            for control_id in self[key].keys():
                json_dict_compatiple[key.value][control_id] = (self[key][control_id][0].value, self[key][control_id][1])
        return json.dumps(json_dict_compatiple)


class CrazyflieInputEvent:
    def __init__(self, cf_input_type: CrazyflieInputs, value: float):
        self.input_type = cf_input_type
        self.value = value


class _CrazyflieControllerCommands:
    STOP_READING_INPUT = 0


class CrazyflieController:
    def __init__(self, input_map: CrazylfieControlsMap):
        self.event_queue = Queue()
        self.input_map = input_map
        self._axes_value_thr = 0.1
        self._vx = 0
        self._vy = 0
        self._vz = 0
        self._z = 0
        self._yaw_rate = 0
        self.max_yaw_rate = 90
        self.max_vx = 1
        self.max_vy = 1
        self.max_vz = 0.5
        self.max_z = 1.3
        self.min_z = 0.3
        self._isFlying = False
        self._last_time = time.time()

    def set_input_map(self, controls_map: CrazylfieControlsMap):
        self.input_map = controls_map

    def map_input(self, event):
        if self.input_map is not None:
            if event.type == pygame.JOYAXISMOTION:
                if event.axis in self.input_map[JoyInputType.AXIS].keys():
                    cf_input_type = self.input_map[JoyInputType.AXIS][event.axis][0]
                    sign = self.input_map[JoyInputType.AXIS][event.axis][1]
                    return CrazyflieInputEvent(cf_input_type, sign*event.value)
            elif event.type == pygame.JOYBUTTONUP:
                if event.button in self.input_map[JoyInputType.BUTTON].keys():
                    return CrazyflieInputEvent(self.input_map[JoyInputType.BUTTON][event.button][0], 0)
            elif event.type == pygame.JOYBUTTONDOWN:
                if event.button in self.input_map[JoyInputType.BUTTON].keys():
                    return CrazyflieInputEvent(self.input_map[JoyInputType.BUTTON][event.button][0], 1)
        return None

    def unassign_crazyflie_input(self, cf_input_type: CrazyflieInputs):
        controls_to_remove = []
        for control_input_type in self.input_map.keys():
            for control in self.input_map[control_input_type].keys():
                if self.input_map[control_input_type][control][0] == cf_input_type:
                    controls_to_remove.append((control_input_type, control))
        for inp_type, control in controls_to_remove:
            self.input_map[inp_type].pop(control)

    def assign_map(self, event, cf_input_type: CrazyflieInputs):
        if event.type == pygame.JOYAXISMOTION:
            if abs(event.value) > 0.8:
                self.input_map[JoyInputType.AXIS][event.axis] = (cf_input_type, np.sign(event.value))
                return True
        elif event.type == pygame.JOYBUTTONUP:
            self.input_map[JoyInputType.BUTTON][event.button] = (cf_input_type, 1)
            return True
        return False

    def control_task(self, cf):
        dt = time.time() - self._last_time
        self._last_time += dt
        if not self.event_queue.empty():
            e = self.map_input(self.event_queue.get(block=False))  # type: CrazyflieInputEvent
            if e is not None:
                if e.input_type == CrazyflieInputs.ASSISTED_MODE and e.value == 0:
                    print("Assist button released")
                    if cf is not None:
                        cf.commander.send_stop_setpoint()
                    self._isFlying = False
                elif e.input_type == CrazyflieInputs.ASSISTED_MODE and e.value == 1:
                    print("Assist button clicked")
                    self._isFlying = True
                    self._z = 0.3
                    self._vx = 0
                    self._vy = 0
                    if cf is not None:
                        cf.param.set_value('kalman.resetEstimation', '1')
                    time.sleep(1)
                    if cf is not None:
                        pass
                        cf.commander.send_hover_setpoint(vx=0, vy=0, yawrate=0, zdistance=self._z)
                    self._last_time = time.time()
                    dt = 0
                elif e.input_type == CrazyflieInputs.YAW:
                    if abs(e.value) > self._axes_value_thr:
                        # print("YAW={}".format(e.value))
                        self._yaw_rate = e.value * self.max_yaw_rate
                    else:
                        self._yaw_rate = 0
                elif e.input_type == CrazyflieInputs.PITCH:
                    if abs(e.value) > self._axes_value_thr:
                        # print("PITCH={}".format(e.value))
                        self._vx = e.value * self.max_vx
                    else:
                        self._vx = 0
                elif e.input_type == CrazyflieInputs.ROLL:
                    if abs(e.value) > self._axes_value_thr:
                        # print("ROLL={}".format(e.value))
                        self._vy = e.value * self.max_vy
                    else:
                        self._vy = 0
                elif e.input_type == CrazyflieInputs.THRUST:
                    if abs(e.value) > self._axes_value_thr:
                        # print("THRUST={}".format(e.value))
                        self._vz = e.value * self.max_vz
                    else:
                        self._vz = 0
        if self._isFlying:
            if self.min_z <= self._z <= self.max_z:
                self._z += self._vz * dt
            if self._z < self.min_z:
                self._z = self.min_z
            elif self._z > self.max_z:
                self._z = self.max_z
            if cf is not None:
                pass
                cf.commander.send_hover_setpoint(vx=self._vx, vy=self._vy, yawrate=self._yaw_rate, zdistance=self._z)


class CrazyflieControllerManager:
    pygame.init()
    pygame.joystick.init()
    _selected_controller = None  # type: Joystick
    _is_reading_input = False
    _callbacks_to_notify_on_selected_controller_change = []
    _input_received_queues = []
    _command_queue = Queue()
    _read_process = None
    _registered_callbacks = []
    _input_read_queue = Queue()
    _callback_thread_running = False
    _input_read_thread = Thread()

    @classmethod
    def register_callback(cls, callback):
        cls._registered_callbacks.append(callback)
        cls.add_queue_to_read_process(cls._input_read_queue)
        if not cls._input_read_thread.is_alive():
            cls._callback_thread_running = True
            cls._input_read_thread = Thread(target=cls._input_read_thread_task, daemon=True)
            cls._input_read_thread.start()

    @classmethod
    def unregister_callback(cls, callback):
        cls._registered_callbacks = list(filter(lambda x: x != callback, cls._registered_callbacks))
        if len(cls._registered_callbacks) == 0:
            cls._callback_thread_running = False
            cls._input_read_thread.join()
            cls.remove_queue_from_read_process(cls._input_read_queue)


    @classmethod
    def _input_read_thread_task(cls):
        while cls._callback_thread_running:
            event = cls._input_read_queue.get(block=False) if not cls._input_read_queue.empty() else None
            if event is not None:
                for clb in cls._registered_callbacks:
                    clb(event)

    @classmethod
    def start_reading_joystick(cls):
        if cls._selected_controller is not None:
            if cls._read_process is not None:
                try:
                    cls._read_process.close()
                except:
                    pass
            cls._read_process = Process(target=cls._input_read_task,
                                        args=(cls._selected_controller,
                                              cls._command_queue,
                                              cls._input_received_queues),
                                        daemon=True)
            cls._read_process.start()
            cls.is_reading_input = True
        else:
            print("No controller selected!")

    @classmethod
    def stop_reading_joystick(cls):
        if cls._read_process is not None and cls._read_process.is_alive():
            cls._command_queue.put(_CrazyflieControllerCommands.STOP_READING_INPUT)
        cls._is_reading_input = False

    @classmethod
    def add_queue_to_read_process(cls, queue: Queue):
        cls.stop_reading_joystick()
        if queue not in cls._input_received_queues:
            cls._input_received_queues.append(queue)
        cls.start_reading_joystick()

    @classmethod
    def remove_queue_from_read_process(cls, queue: Queue, start_reading=True):
        cls.stop_reading_joystick()
        cls._input_received_queues = list(filter(lambda x: x != queue, cls._input_received_queues))
        if len(cls._input_received_queues) != 0 and start_reading:
            cls.start_reading_joystick()

    @staticmethod
    def _input_read_task(joystick: Joystick, cmd_queue: Queue, input_received_queues: List[Queue]):
        try:
            joystick.init()
        except Exception:
            print("Couldn't initialize joystick")
        cmd = None
        while cmd != _CrazyflieControllerCommands.STOP_READING_INPUT:
            js_events = pygame.event.get()
            try:
                cmd = cmd_queue.get(block=False) if not cmd_queue.empty() else None
            except:
                cmd = None
            for e in js_events:
                if hasattr(e, 'instance_id'):
                    if e.instance_id == joystick.get_instance_id():
                        for queue in input_received_queues:
                            if queue.qsize() < 20:
                                queue.put(ControllerInputEvent(e))

    @staticmethod
    def get_available_controllers():
        return [pygame.joystick.Joystick(x) for x in range(pygame.joystick.get_count())]

    @classmethod
    def set_selected_controller(cls, selected_controller: Joystick):
        if not cls._is_reading_input:
            try:
                selected_controller.init()
            except:
                print("ERROR: Couldn't initialize joystick. Did you connect it?")
            cls._selected_controller = selected_controller
            for callback in cls._callbacks_to_notify_on_selected_controller_change:
                callback(selected_controller)
        else:
            cls.stop_reading_joystick()
            cls._selected_controller = selected_controller
            cls.start_reading_joystick()
            for callback in cls._callbacks_to_notify_on_selected_controller_change:
                callback(selected_controller)

    @classmethod
    def set_selected_controller_by_id(cls, id: int):
        cls.set_selected_controller(pygame.joystick.Joystick(id))

    @classmethod
    def get_selected_controller(cls) -> Joystick:
        return cls._selected_controller

    @classmethod
    def quit(cls):
        cls._callback_thread_running = False
        if cls._input_read_thread is not None and cls._input_read_thread.is_alive():
            cls._input_read_thread.join()
        cls.stop_reading_joystick()
        pygame.quit()


if __name__ == "__main__":
    import numpy as np
    from cflib.positioning.position_hl_commander import PositionHlCommander
    # from cflib.crazyflie import Crazyflie
    # from cflib.crazyflie.syncCrazyflie import SyncCrazyflie
    #
    # cflib.crtp.init_drivers()
    #
    # with SyncCrazyflie(link_uri='radio://0/80/2M/E7E7E7E7E7', cf=Crazyflie(rw_cache='./cache')) as scf:
    #     controller = CrazyflieController()
    #     id = 0
    #     selected_joystick = controller.available_joysticks[id]
    #     print(selected_joystick.get_name())
    #     input_map = CrazylfieControlsMapping(selected_joystick)
    #     controller.set_input_mapping(input_map)
    #     controller.start_reading_joystick(id)
    #     while controller.map_task():
    #         pass



