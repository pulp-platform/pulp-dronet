/*-----------------------------------------------------------------------------
 Copyright (C) 2024 University of Bologna, Italy, ETH Zurich, Switzerland.
 All rights reserved.

 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 See LICENSE in the top directory for details.
 You may obtain a copy of the License at

   http://www.apache.org/licenses/LICENSE-2.0

 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.

 File:    frame_streamer.h.c
 Author:  Lorenzo Lamberti      <lorenzo.lamberti@unibo.it>
 Date:    01.03.2024
-------------------------------------------------------------------------------*/

#ifndef __TOOLS__FRAME_STREAMER_H__
#define __TOOLS__FRAME_STREAMER_H__

#include <bsp/buffer.h>
#include <tools/services.h>
#include <tools/frame_streamer_constants.h>
#include "gaplib/jpeg_encoder.h"
#include "bsp/transport.h"

struct frame_streamer_conf {
  struct pi_device *transport;
  frame_streamer_format_e format;
  uint32_t width;
  uint32_t height;
  uint32_t depth;
  char *name;
};

// ===================Added from frame_streamer.c =====================
typedef struct
{
  jpeg_encoder_t encoder;
  pi_buffer_t bitstream;
} frame_streamer_jpeg_t;

typedef struct
{
  struct pi_transport_header header;
  frame_streamer_open_req_t req;
} frame_streamer_open_req_full_t;


struct frame_streamer_s {
  struct pi_device *transport;
  frame_streamer_format_e format;
  int channel;
  struct pi_transport_header header;
  frame_streamer_open_req_full_t req;
  frame_streamer_jpeg_t *jpeg;
  unsigned int height;
  unsigned int width;
};

// ===================Added from frame_streamer.c =====================

typedef struct frame_streamer_s frame_streamer_t;


int frame_streamer_conf_init(struct frame_streamer_conf *conf);

frame_streamer_t *frame_streamer_open(struct frame_streamer_conf *conf);

int frame_streamer_send_async(frame_streamer_t *streamer, pi_buffer_t *buffer, pi_task_t *task);

int frame_streamer_send(frame_streamer_t *streamer, pi_buffer_t *buffer);

#endif