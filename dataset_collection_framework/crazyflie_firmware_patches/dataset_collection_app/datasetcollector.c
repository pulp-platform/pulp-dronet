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

 File:    app_dronet.c
 Author:  Lorenzo Lamberti      <lorenzo.lamberti@unibo.it>
          Daniel Rieben		    <riebend@student.ethz.ch>
 Date:    01.03.2024
-------------------------------------------------------------------------------*/

#define DEBUG_MODULE "datasetCollectorDriver"
#include "debug.h"
#include "deck.h"
#include "FreeRTOS.h"
#include "task.h"
#include "log.h"
#include "deck_constants.h"
#include "usec_time.h"
#include <math.h>
#include "datasetcollector.h"

// Defines
#define PIN_CLK_RESET DECK_GPIO_TX1     // Pin 3
#define SIGNAL_PERIOD 100 //ms


// My Variables
static bool tick_sync_done = false;
static unsigned int gap_ticks = 0;
static int gap_clock_state = 0;

// My Functions
static void tickSyncSequence(void);


void tickSyncSequence(void)
{
  if (!tick_sync_done){
    vTaskDelay(M2T(3000));  // Wait for 3 sec to be sure that the gap8 is ready
    // generate reset signal on GPIO (it a HIGH->LOW impulse)
    digitalWrite(PIN_CLK_RESET, HIGH);  // clock reset signal
    vTaskDelay(M2T(SIGNAL_PERIOD));
    digitalWrite(PIN_CLK_RESET, LOW);
    while(digitalRead(PIN_CLK_RESET) == HIGH);
    //reset the ticks
    gap_ticks = 0;

    tick_sync_done = true;
  }


}

unsigned int get_dataset_collector_timestamp(){
    return gap_ticks;
}

static void dataCollectorTask(){
    systemWaitStart();
    vTaskDelay(M2T(500));
    tickSyncSequence();
    while(1){
        int current_state = digitalRead(DECK_GPIO_RX1);
        if(gap_clock_state != current_state){
            gap_clock_state = current_state;
            gap_ticks++;
        }
    }
}


static void datasetCollectorInit()
{
    // STM32: drives PIN_CLK_RESET LOW->HIGH once as an initial synchronization signal (read by GAP8)
    // GAP8: drives DECK_GPIO_RX1 continuosly HIGH->LOW and LOW->HIGH for periodic synchronization (read by the STM32). This is a workaround because GPIO interrupts dont work properly on GAP8 right now.

    // Setup the GPIO and write it to 0
    pinMode(PIN_CLK_RESET, OUTPUT);  // init GPIO mode
    digitalWrite(PIN_CLK_RESET, LOW); //init value to LOW
    vTaskDelay(10);

    pinMode(DECK_GPIO_RX1, INPUT);  // init GPIO mode
    gap_clock_state = digitalRead(DECK_GPIO_RX1);

    DEBUG_PRINT("Initialize dataset collector driver!\n");
    initUsecTimer();
    setCustomTimestamp(get_dataset_collector_timestamp);

    xTaskCreate(dataCollectorTask, "dataset-collector-task", 2*configMINIMAL_STACK_SIZE, NULL, 1, NULL);
}


static bool datasetCollectorTest()
{
  DEBUG_PRINT("Test dataset collector driver!\n");
  return true;
}

static const DeckDriver dataset_collector_Driver = {
  .name = "datasetCollectorDriver",
  .init = datasetCollectorInit,
  .test = datasetCollectorTest,
};

DECK_DRIVER(dataset_collector_Driver);

