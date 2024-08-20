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
# File:    CrazyflieCameraStreamer.py                                         #
# Authors:                                                                    #
#          Lorenzo Lamberti <lorenzo.lamberti@unibo.it>                       #
#          Daniel Rieben		    <riebend@student.ethz.ch>                 #
# Date:    01.03.2024                                                         #
#-----------------------------------------------------------------------------#

from multiprocessing import Process, Queue
import socket
from PIL import Image, ImageFile
import io
import struct
from typing import *
from threading import Thread

ImageFile.LOAD_TRUNCATED_IMAGES = True


class CrazyflieCameraStreamer:
    CMD_START_STREAMING = 0
    CMD_STOP_STREAMING = 1
    CMD_DISCONNECT = 2
    CMD_CONNECT = 3
    CMD_SHUTDOWN = 4

    def __init__(self, deck_ip="192.168.4.2", deck_port=5000):
        self._image_queue = Queue()
        self._request_queue = Queue()
        self.deck_ip = deck_ip
        self.deck_port = deck_port
        self._streamer_task = None
        self.image_receive_thread = None
        self.next_cmd = None
        self.next_cmd_data = None
        self.read_next = False
        self.client_socket = None
        self.receive_image_run = False
        self.is_connected = False

    def image_receive_task(self):
        imgdata = bytearray()
        wait_for_new_image = True
        while self.receive_image_run:
            strng = self.client_socket.recv(512)

            # Look for start-of-frame
            start_idx = strng.find(b"\xff\xd8")

            # Concatenate image data, once finished send it to the UI
            if start_idx >= 0:
                if wait_for_new_image:
                    imgdata = strng[start_idx:]
                    wait_for_new_image = False
                else:
                    imgdata_complete = imgdata + strng[:start_idx]
                    time_stamp = struct.unpack('<Q', imgdata_complete[-8:])[0]
                    imgdata_complete = imgdata_complete[:-8]
                    end_idx = imgdata_complete.find(b"\xff\xd9")
                    imgdata_complete = imgdata_complete[:end_idx] + imgdata_complete[end_idx + 2:] + b"\xff\xd9"
                    try:
                        picture_stream = io.BytesIO(imgdata_complete)
                        picture = Image.open(picture_stream, formats=['JPEG'])
                        self._image_queue.put((time_stamp, picture))
                    except:
                        print('Could not open jpeg')
                    imgdata = strng[start_idx:]
            else:
                imgdata += strng


    @staticmethod
    def stream_images_task(streamer):
        cmd = None
        while cmd != CrazyflieCameraStreamer.CMD_SHUTDOWN:
            cmd, data = streamer._request_queue.get(block=False) if not streamer._request_queue.empty() else [None, None]
            if cmd == CrazyflieCameraStreamer.CMD_CONNECT:
                print("Connecting to socket on {}:{}...".format(data["ip"], data["port"]))
                streamer.client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                streamer.client_socket.settimeout(1)
                try:
                    streamer.client_socket.connect((data["ip"], data["port"]))
                    streamer.is_connected = True
                    print("Connected to socket!")
                except Exception as excp:
                    streamer.is_connected = False
                    print("Could not connect!")
            elif cmd == CrazyflieCameraStreamer.CMD_DISCONNECT:
                try:
                    streamer.receive_image_run = False
                    streamer.client_socket.shutdown(1)
                    streamer.is_connected = False
                except:
                    print("Wasn't able to close the socket!")
                    pass
            elif cmd == CrazyflieCameraStreamer.CMD_START_STREAMING:
                if streamer.is_connected:
                    streamer.receive_image_run = True
                    streamer.image_receive_thread = Thread(target=streamer.image_receive_task, daemon=True)
                    streamer.image_receive_thread.start()
            elif cmd == CrazyflieCameraStreamer.CMD_STOP_STREAMING:
                streamer.receive_image_run = False

        try:
            streamer.receive_image_run = False
            streamer.client_socket.shutdown(1)
            streamer.is_connected = False
        except:
            print("Wasn't able to close the socket!")
            pass

    def _put_to_request_queue(self, cmd, data=None):
        self._request_queue.put((cmd, data))

    def get_image_in_queue(self):
        return self._image_queue.get() if not self._image_queue.empty() else None

    def connect(self, deck_ip: AnyStr, deck_port: int, start_streaming: bool = False):
        self.deck_ip = deck_ip
        self.deck_port = deck_port
        if self._streamer_task is None:
            self._streamer_task = Process(target=self.stream_images_task, args=(self,))
            self._streamer_task.daemon = True
            self._streamer_task.start()
        self._put_to_request_queue(CrazyflieCameraStreamer.CMD_CONNECT, data={"ip": deck_ip, "port": deck_port})

    def start_streaming(self):
        self._put_to_request_queue(CrazyflieCameraStreamer.CMD_START_STREAMING)

    def disconnect(self):
        self._put_to_request_queue(CrazyflieCameraStreamer.CMD_DISCONNECT)

    def stop_streaming(self):
        self._put_to_request_queue(CrazyflieCameraStreamer.CMD_STOP_STREAMING)

    def shutdown(self):
        self._put_to_request_queue(CrazyflieCameraStreamer.CMD_SHUTDOWN)


if __name__ == "__main__":
    image_stream = CrazyflieCameraStreamer()
    try:
        while True:
            image_stream.stream_images_task(image_stream)
    except KeyboardInterrupt:
        pass

