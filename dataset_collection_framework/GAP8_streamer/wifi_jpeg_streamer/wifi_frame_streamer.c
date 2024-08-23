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

 File:    wifi_jpeg_streamer.c
 Author:  Daniele Palossi <dpalossi@iis.ee.ethz.ch>
          Lorenzo Lamberti      <lorenzo.lamberti@unibo.it>
 Date:    01.03.2024
-------------------------------------------------------------------------------*/

#include "bsp/camera/himax.h"
#include "bsp/transport/nina_w10.h"
#include "stdio.h"
#include "pmsis.h"
#include "../inc/frame_streamer.h" //temporary patch

// #define VERBOSE

#ifdef VERBOSE
#define debug_printf printf
#else
#define debug_printf
#endif

#define CAM_WIDTH   324
#define CAM_HEIGHT  244

uint64_t time_ticks;
static unsigned char *image;

// LED task
static struct pi_device led;
static pi_task_t led_task;
static int led_val = 0;

// CAMERA task
static struct pi_device camera;
static pi_task_t cam_task;

// STREAMER task
static pi_buffer_t buffer;
static pi_task_t stream_task;
static frame_streamer_t *streamer;
static char *name = "video_stream";

// WiFi
static struct pi_device wifi;

// GPIO interrupt
static struct pi_device gpio;
static int gpio_val = 0;
static pi_task_t gpio_task;
static pi_gpio_callback_t cb_gpio = {0};

// Toggle timer
static rt_timer_t timer;

static void led_handler();
static void gpio_toggle_handler();
static void streamer_handler();
static void camera_handler();
static void polling_synch();
static void gpio_handler();
static void init_memory();
static void init_camera();
static void init_wifi();
static void init_streamer();
static void init_led();
static void init_gpio();
static void init_gpio_toggle_timer();


int frame_streamer_send_async_with_timestamp(frame_streamer_t *streamer, pi_buffer_t *buffer, uint64_t* image_timestamp,  pi_task_t *task)
{
  uint8_t *frame = buffer->data;

  if (streamer->format == FRAME_STREAMER_FORMAT_RAW)
  {
    int size = pi_buffer_size(buffer);
    if (pi_transport_send_header(streamer->transport, &streamer->header, streamer->channel, size))
      return -1;
    if (pi_transport_send_async(streamer->transport, buffer->data, size, task))
      return -1;
  }

  if (streamer->format == FRAME_STREAMER_FORMAT_JPEG)
  {
    uint32_t size;
    uint32_t tot_size=0;
    while(1)
    {
      // We encode part of the image for each while(1) iteration.
      //    end==1 means that we didnt finish the frame TX.
      //    end==0 means that we are sending the last part of the image and we are finishing the frame TX.
      int end = jpeg_encoder_process(&streamer->jpeg->encoder, buffer, &streamer->jpeg->bitstream, &size);
      streamer->header.info = end ? 0 : 1;

      if (end == 0) { //last packet of the frame. So we attach the timestamp!
        *((uint64_t *)(streamer->jpeg->bitstream.data+size)) = *((uint64_t *)image_timestamp);
        size=size+8;
      }
      tot_size +=size;

#ifdef VERBOSE
      printf("end: %d \t packet size: %u \n", end, size);
#endif

      if (pi_transport_send_header(streamer->transport, &streamer->header, streamer->channel, size))
        return -1;
      if (pi_transport_send(streamer->transport, streamer->jpeg->bitstream.data, size))
        return -1;

      if (end == 0) {
#ifdef VERBOSE
        printf("image size %d , acquisition timestamp: %10llu uSec.\n", tot_size, *image_timestamp);
#endif
        break;
      }
    }
    pi_task_push(task);
  }
  else
  {
    return -1;
  }
  return 0;
}

static void led_handler() {

    led_val ^= 1;
    pi_gpio_pin_write(&led, PI_GPIO_A2_PAD_14_A2, led_val);
    pi_task_push_delayed_us(pi_task_callback(&led_task, led_handler, NULL), 500000);
}

static void gpio_toggle_handler() {
    time_ticks++;
    gpio_val ^= 1;
    pi_gpio_pin_write(&gpio, PI_GPIO_A25_PAD_39_A7, gpio_val);
    pi_task_push_delayed_us(pi_task_callback(&gpio_task, gpio_toggle_handler, NULL), 15000);
}

static void streamer_handler() {

    pi_camera_control(&camera, PI_CAMERA_CMD_STOP, 0);

    // frame_streamer_send_async(streamer, &buffer, pi_task_callback(&cam_task, camera_handler, NULL));
      frame_streamer_send_async_with_timestamp(streamer, &buffer, &time_ticks , pi_task_callback(&cam_task, camera_handler, NULL)); // my frame streamer adds timestamp
#ifdef VERBOSE
  printf("------------------------------------------------------\n");
#endif

}


static void camera_handler() {

    pi_camera_capture_async(&camera, image, CAM_WIDTH*CAM_HEIGHT, pi_task_callback(&stream_task, streamer_handler, NULL));

    pi_camera_control(&camera, PI_CAMERA_CMD_START, 0);

}




static void init_memory() {

    image = (unsigned char *)pi_l2_malloc(CAM_WIDTH*CAM_HEIGHT*sizeof(unsigned char));

#ifdef VERBOSE
    printf("L2 Image alloc\t%dB\t@ 0x%08X:\t%s\n", CAM_WIDTH*CAM_HEIGHT*sizeof(unsigned char), (unsigned char*) image, image?"Ok":"Failed");
#endif

    if(image == NULL) pmsis_exit(-1);
}


static void init_camera() {

    int32_t errors = 0;
    uint8_t set_value = 3;
    struct pi_himax_conf cam_conf;

    pi_himax_conf_init(&cam_conf);

    cam_conf.format = PI_CAMERA_QVGA;

    pi_open_from_conf(&camera, &cam_conf);

    errors = pi_camera_open(&camera);

#ifdef VERBOSE
    printf("HiMax camera init:\t\t\t%s\n", errors?"Failed":"Ok");
#endif

    if(errors) pmsis_exit(errors);

    // image rotation
    pi_camera_reg_set(&camera, IMG_ORIENTATION, &set_value);
    pi_camera_control(&camera, PI_CAMERA_CMD_AEG_INIT, 0);
}


static void init_wifi() {

    int32_t errors = 0;
    struct pi_nina_w10_conf nina_conf;

    pi_nina_w10_conf_init(&nina_conf);

    nina_conf.ssid = "";
    nina_conf.passwd = "";
    nina_conf.ip_addr = "0.0.0.0";
    nina_conf.port = 5555;

    pi_open_from_conf(&wifi, &nina_conf);

    errors = pi_transport_open(&wifi);

#ifdef VERBOSE
    printf("NINA WiFi init:\t\t\t\t%s\n", errors?"Failed":"Ok");
#endif

    if(errors) pmsis_exit(errors);
}


static void init_streamer() {

    struct frame_streamer_conf streamer_conf;

    frame_streamer_conf_init(&streamer_conf);

    streamer_conf.transport = &wifi;
    streamer_conf.format = FRAME_STREAMER_FORMAT_JPEG;
    streamer_conf.width = CAM_WIDTH;
    streamer_conf.height = CAM_HEIGHT;
    streamer_conf.depth = 1;
    streamer_conf.name = name;

    streamer = frame_streamer_open(&streamer_conf);

    pi_buffer_init(&buffer, PI_BUFFER_TYPE_L2, image);
    pi_buffer_set_format(&buffer, CAM_WIDTH, CAM_HEIGHT, 1, PI_BUFFER_FORMAT_GRAY);

#ifdef VERBOSE
    printf("Streamer init:\t\t\t\t%s\n", streamer?"Ok":"Failed");
#endif

    if(streamer == NULL) pmsis_exit(-1);
}


static void init_led() {

    int32_t errors = 0;
    struct pi_gpio_conf led_conf = {0};

    pi_gpio_conf_init(&led_conf);
    pi_open_from_conf(&led, &led_conf);

    errors = pi_gpio_open(&led);

#ifdef VERBOSE
    printf("LED GPIO init:\t\t\t\t%s\n", errors?"Failed":"Ok");
#endif

    if(errors) pmsis_exit(errors);

    pi_gpio_pin_configure(&led, PI_GPIO_A2_PAD_14_A2, PI_GPIO_OUTPUT);
}


static void gpio_handler() {

    printf(" --- CALLBACK --- \n");
}

static void polling_synch() {
    uint32_t value = 0;
#ifdef VERBOSE
    debug_printf("Start polling for synch\n");
#endif
    while(value == 0){
        pi_gpio_pin_read(&gpio, PI_GPIO_A24_PAD_38_B6, &value);
        if(value == 1){
            break;
        }
    }
    while(value == 1){
        pi_gpio_pin_read(&gpio, PI_GPIO_A24_PAD_38_B6, &value);
        if(value == 0){
            time_ticks = 0;
            break;
        }
    }
#ifdef VERBOSE
        debug_printf("GPIO value: %i\n", value);
#endif
#ifdef VERBOSE
    // debug_printf("Start time: %llu us\n", start_time_us);
#endif
}


static void init_gpio() {

    int32_t errors = 0;
    struct pi_gpio_conf gpio_conf = {0};
    pi_gpio_e gpio_in = PI_GPIO_A24_PAD_38_B6;
//  pi_gpio_notif_e irq_type = PI_GPIO_NOTIF_RISE;
    pi_gpio_notif_e irq_type = PI_GPIO_NOTIF_NONE;  // For now don't add interrupt
    pi_gpio_flags_e cfg_flags = PI_GPIO_INPUT | PI_GPIO_PULL_DISABLE | PI_GPIO_DRIVE_STRENGTH_LOW;
    int32_t gpio_mask = (1 << (gpio_in & PI_GPIO_NUM_MASK));

    pi_gpio_conf_init(&gpio_conf);
    pi_open_from_conf(&gpio, &gpio_conf);

    errors = pi_gpio_open(&gpio);

#ifdef VERBOSE
    printf("Callback GPIO init:\t\t\t%s\n", errors?"Failed":"Ok");
#endif

    if(errors) pmsis_exit(errors);

    pi_gpio_pin_configure(&gpio, gpio_in, cfg_flags);
    pi_gpio_pin_notif_configure(&gpio, gpio_in, irq_type);

    // pi_gpio_callback_init(&cb_gpio, gpio_mask, gpio_handler, NULL);  //Only needed if interrupt used

    pi_gpio_e gpio_out = PI_GPIO_A25_PAD_39_A7;
    pi_gpio_flags_e cfg_flags_out = PI_GPIO_OUTPUT | PI_GPIO_PULL_ENABLE | PI_GPIO_DRIVE_STRENGTH_HIGH;
    pi_gpio_pin_configure(&gpio, gpio_out, cfg_flags_out);

    pi_gpio_pin_write(&gpio, PI_GPIO_A25_PAD_39_A7, gpio_val);

}


int main_task(void) {

    int32_t errors = 0;

#ifdef VERBOSE
    printf("Entering main controller...\n");
#endif

    pi_freq_set(PI_FREQ_DOMAIN_FC, 250000000);

    // Image L2 allocation
    init_memory();


    // Camera initialization
    init_camera();

    // GPIO callback initialization
    // init_gpio();

    // Wifi initialization
    init_wifi();

    // Streamer initialization
    init_streamer();

#ifdef VERBOSE
    printf("Task LED started:\t\t\tOk\n");
#endif

    // Initialize GPIO's
    init_gpio();

    // Polling timestamp synchronization
    polling_synch();

    // Blinking LED initialization
    init_led();

    // Start task: led blinking
    pi_task_push(pi_task_callback(&led_task, led_handler, NULL));

    // Start task: GPIO periodic signal for synch
    pi_task_push(pi_task_callback(&gpio_task, gpio_toggle_handler, NULL));

    // Start task: camera capture
    pi_task_push(pi_task_callback(&cam_task, camera_handler, NULL));
#ifdef VERBOSE
    printf("Task Capture started:\t\t\tOk\n");
#endif

    while(1) {
        pi_yield();
    }

    pi_l2_free(image, CAM_WIDTH*CAM_HEIGHT*sizeof(unsigned char));
    pi_gpio_close(&led);
    pi_gpio_close(&gpio);
    pmsis_exit(errors);

    return 0;
}


/* Program Entry. */
int main(void) {

#ifdef VERBOSE
    printf("\n\n\t *** PMSIS Kickoff trasmission ***\n\n");
#endif
    return pmsis_kickoff((int *) main_task);
}
