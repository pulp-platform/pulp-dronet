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
          Vlad Niculescu 	    <vladn@iis.ee.ethz.ch>
 Date:    01.03.2024
-------------------------------------------------------------------------------*/

#include <string.h>
#include <stdint.h>
#include <stdbool.h>

#include "app.h"
#include "FreeRTOS.h"
#include "system.h"
#include "task.h"
#include "debug.h"
#include "stabilizer_types.h"
#include "estimator_kalman.h"
#include "commander.h"
#include "log.h"
#include "param.h"
#include <math.h>
#include "uart_dma_setup.h"
#include "config_dronet.h"

#define BUFFERSIZE 8

/* --------------- GUI PARAMETERS --------------- */
// Global variables for the parameters
float max_forward_index = MAX_FORWARD_INDEX;
float alpha_vel = ALPHA_VEL;
float alpha_yaw = ALPHA_YAW;
float yaw_scaling = YAW_SCALING;
float flying_height = PULP_TARGET_H;
float integral_weight = INTEGRAL_WEIGHT;
float integral_thresh = INTEGRAL_THRESH;
float critical_prob_collision = CRITICAL_PROB_COLL;
float nemo_quantum = NEMO_QUANTUM;
// My parameters for enabling/disabling some parts of code. 1=Active, 0=Non active
uint8_t enable_integral = ENABLE_INTEGRAL;
uint8_t enable_low_pass = ENABLE_LOW_PASS;
uint8_t quadratic = ENABLE_QUADRATIC;
uint8_t emergency_br = ENABLE_EMERGENCY_BR;
uint8_t debug = 0; 								// activate debug prints

// START / STOP mission parameter
uint8_t fly = 0; 		// Takeoff/landing command (GUI parameter)
uint8_t landed = 0; 	// Flag for indicating whether the drone landed

/* --------------- GLOBAL VARIABLES --------------- */
// CNN output
int8_t pulpRxBuffer[BUFFERSIZE];
int32_t cnn_data_int[BUFFERSIZE/4];
float cnn_data_float[2];
volatile uint8_t dma_flag = 0;
// Processed data
float steering_angle, prob_of_col;
float forward_velocity = 0.0f, forward_velocity_old = 0.0f;
float angular_velocity = 0.0f, angular_velocity_old = 0.0f;
float prob_integral = 0.0f;
setpoint_t setp_dronet;


/* --------------- FUNCTION DEFINITION --------------- */
void process_cnn_output(int32_t* cnn_output_int, float* cnn_output_float);
double sigmoid(float x);
void land(void);
void takeoff(float height);
void headToPosition(float x, float y, float z, float yaw);
static setpoint_t create_setpoint(float x_vel, float z, float yaw_rate);
void compute_angular_velocity(float steering_angle, float* angular_velocity);
void compute_forward_velocity(float prob_of_col, float* forward_velocity_target);
float low_pass_filtering(float data_new, float data_old, float alpha);
void emergency_brake(float* forward_velocity, float critical_prob_collision);
float integrate_collision(float prob_of_col, float integral_weight);


/* --------------- FUNCTIONS --------------- */
// Fly forward functions
static setpoint_t create_setpoint(float x_vel, float z, float yaw_rate)
{
    setpoint_t setpoint;
    memset(&setpoint, 0, sizeof(setpoint_t));
    setpoint.mode.x = modeVelocity;
    setpoint.mode.y = modeVelocity;
    setpoint.mode.z = modeAbs;
    setpoint.mode.yaw = modeVelocity;

    setpoint.velocity.x	= x_vel;
    setpoint.velocity.y	= 0.0f;
    setpoint.position.z = z;
    setpoint.attitudeRate.yaw = yaw_rate;
    setpoint.velocity_body = true;
    return setpoint;
}

void headToPosition(float x, float y, float z, float yaw)
{
    setpoint_t setpoint;
    memset(&setpoint, 0, sizeof(setpoint_t));

    setpoint.mode.x = modeAbs;
    setpoint.mode.y = modeAbs;
    setpoint.mode.z = modeAbs;
    setpoint.mode.yaw = modeAbs;

    setpoint.position.x = x;
    setpoint.position.y = y;
    setpoint.position.z = z;
    setpoint.attitude.yaw = yaw;
    commanderSetSetpoint(&setpoint, 3);
}

// TAKEOFF and LANDING FUNCTIONS
void takeoff(float height)
{
    point_t pos;
    memset(&pos, 0, sizeof(pos));
    estimatorKalmanGetEstimatedPos(&pos);

    // first step: taking off gradually, from a starting height of 0.2 to the desired height
    int endheight = (int)(100*(height-0.2f));
    for(int i=0; i<endheight; i++)
    {
        headToPosition(pos.x, pos.y, 0.2f + (float)i / 100.0f, 0);
        vTaskDelay(50);
    }
    // keep constant height
    for(int i=0; i<100; i++)
    {
        headToPosition(pos.x, pos.y, height, 0);
        vTaskDelay(50);
    }
}

void land(void)
{
    point_t pos;
    memset(&pos, 0, sizeof(pos));
    estimatorKalmanGetEstimatedPos(&pos);

    float height = pos.z;
    float current_yaw = logGetFloat(logGetVarId("stateEstimate", "yaw"));

    for(int i=(int)100*height; i>100*FINAL_LANDING_HEIGHT; i--) {
        headToPosition(pos.x, pos.y, (float)i / 100.0f, current_yaw);
        vTaskDelay(20);
    }
    vTaskDelay(200);
}

// CNN POST-PROCESSING
double sigmoid(float x)
{
     float result;
     result = 1 / (1 + expf(-x));
     return result;
}

void process_cnn_output(int32_t* cnn_output_int, float* cnn_output_float)
{
    // [0]=steering
    cnn_output_float[0] = (float) (cnn_output_int[0] * nemo_quantum);
    // [1]=collision
    cnn_output_float[1] = (float) (cnn_output_int[1] * nemo_quantum);

    if(cnn_output_float[0] < -1.0f) cnn_output_float[0]	= -1.0f;
    if(cnn_output_float[0] > 1.0f) cnn_output_float[0] 	= 1.0f;
    // if(cnn_output_float[1] < 0.1f) cnn_output_float[1] 	= 0.0f;
}

float integrate_collision(float prob_of_col, float integral_weight)
{
    // integrate the prob of collision over time if > integral_thresh
    prob_integral += (prob_of_col - integral_thresh); //default: integral_thresh = 0.3

    // limit range to [0,2]. This is arbitrary,
    if (prob_integral < 0.0f) prob_integral = 0.0f;
    if (prob_integral > 2.0f) prob_integral = 2.0f;

    // Add the integrated factor to the raw probability of collision
    prob_of_col = (prob_of_col + integral_weight*prob_integral);
    return prob_of_col;
}

void emergency_brake(float* forward_velocity, float critical_prob_collision)
{
    // Stop if velocity is prob of collision is too high
    if(*forward_velocity < ((1.0f - critical_prob_collision) * max_forward_index))
        *forward_velocity = 0.0f;
}

float low_pass_filtering(float data_new, float data_old, float alpha)
{
    float output;
    // Low pass filter the forward velocity
    output = (1.0f - alpha) * data_new + alpha * data_old;
    return output;
}

void compute_forward_velocity(float prob_of_col, float* forward_velocity_target)
{
    // Compute forward velocity
    if(prob_of_col > 1.0f)
        prob_of_col = 1.0f;

    if (quadratic==1)
        *forward_velocity_target = (prob_of_col*prob_of_col - 2*prob_of_col + 1.0f) * max_forward_index; // (prob_of_col-1)^2 * max_forward_index
    else
        *forward_velocity_target = (1.0f - prob_of_col) * max_forward_index;

    if(*forward_velocity_target < 0.0f) {
        DEBUG_PRINT("Negative forward velocity! Drone will stop!\n");
        *forward_velocity_target  = 0.0f;
    }
}

void compute_angular_velocity(float steering_angle, float* angular_velocity)
{
    *angular_velocity = steering_angle*yaw_scaling;
}

void appMain()
{
    DEBUG_PRINT("Dronet v2 started! \n");
    // USART_DMA_Start(115200, pulpRxBuffer, BUFFERSIZE);
    // vTaskDelay(3000);
    systemWaitStart();
    vTaskDelay(1000);
    USART_DMA_Start(115200, pulpRxBuffer, BUFFERSIZE);

    /* ------------------------- NOT FLYING ------------------------- */
    while(!fly)
    {
        if (dma_flag == 1)
        {
            dma_flag = 0;
            if (debug==1) DEBUG_PRINT("1.UART data: %08x  %08x \n", (unsigned)((uint32_t *)pulpRxBuffer)[0], (unsigned)((uint32_t *)pulpRxBuffer)[1]);
            if (debug==1) DEBUG_PRINT("1.UART data: %ld  %ld \n", ((int32_t *)pulpRxBuffer)[0], ((int32_t *)pulpRxBuffer)[1]);
            cnn_data_int[0] = ((int32_t *)pulpRxBuffer)[0];
            cnn_data_int[1] = ((int32_t *)pulpRxBuffer)[1];
            // if (debug==3) DEBUG_PRINT("UART data: %ld  %ld \n", cnn_data_int[0], cnn_data_int[1]);
            process_cnn_output(cnn_data_int, cnn_data_float);  // get scaled data
            if (debug==2) DEBUG_PRINT("2.FLOAT data: %.3f  %.3f \n", (double)cnn_data_float[0], (double)cnn_data_float[1]);
            cnn_data_float[1] = sigmoid(cnn_data_float[1]); // apply sigmoid
            if (debug==3) DEBUG_PRINT("3.FLOAT+SIG: Steering: %.3f, \tCollision: %.3f \n", (double)cnn_data_float[0], (double)cnn_data_float[1]);

            // Extract steering angle and proability of collision
            steering_angle = cnn_data_float[0];
            prob_of_col = cnn_data_float[1];

            // integrate the prob of collision
            if (enable_integral==1){
                prob_of_col = integrate_collision(prob_of_col, INTEGRAL_WEIGHT);
                if (debug==4) DEBUG_PRINT("4.INTEGRATED: Steering: %.3f, \tCollision (integ): %.3f \n", (double)steering_angle, (double)prob_integral);
            }

            // Calculate forward and anglar velocities
            compute_forward_velocity(prob_of_col, &forward_velocity);
            compute_angular_velocity(steering_angle, &angular_velocity);
            if (debug==5) DEBUG_PRINT("5.VELOCITIES: Angular: %.3f, \tForward: %.3f \n", (double)angular_velocity, (double)forward_velocity);

            // Low pass filtering of forward and angular_velocity
            if (enable_low_pass==1){
                forward_velocity = low_pass_filtering(forward_velocity, forward_velocity_old, alpha_vel);
                angular_velocity = low_pass_filtering(angular_velocity, angular_velocity_old, alpha_yaw);
                if (debug==6) DEBUG_PRINT("6.LOW_PASS: Angular: %.3f, \tForward: %.3f \n", (double)angular_velocity, (double)forward_velocity);
            }

            // emergency brake if prob_collision>threshold
            if (emergency_br==1){
                emergency_brake(&forward_velocity, critical_prob_collision);
                if (debug==7) DEBUG_PRINT("7.Emergency brake: Angular: %.3f, \tForward: %.3f \n", (double)angular_velocity, (double)forward_velocity);
            }

            if (debug==8) DEBUG_PRINT("8.FINAL: Angular: %.3f, \tForward: %.3f \n", (double)angular_velocity, (double)forward_velocity);

            // store i-1 values for the low pass filter
            forward_velocity_old = forward_velocity;
            angular_velocity_old = angular_velocity;
        }
    }

    /* ------------------------- TAKING OFF ------------------------- */

    // reset the estimator before taking off
    estimatorKalmanInit();
    // TAKE OFF
    takeoff(flying_height);

    /* ------------------------ Flight Loop ------------------------ */
    uint8_t timeout_counter = 0;
    while(1) {
        // vTaskDelay(5);
        if (fly==0 && landed==0)//land
        {
            land();
            landed=1;
        }
        if (fly==1 && landed==1) //start flying again
        {
            estimatorKalmanInit();
            takeoff(flying_height);
            landed=0;
        }
        // If new UART data is available
        if (dma_flag == 1)
        {
            dma_flag = 0;  // clear the flag
            timeout_counter = 0;
            cnn_data_int[0] = ((int32_t *)pulpRxBuffer)[0];
            cnn_data_int[1] = ((int32_t *)pulpRxBuffer)[1];
            process_cnn_output(cnn_data_int, cnn_data_float);  // get scaled data
            cnn_data_float[1] = sigmoid(cnn_data_float[1]); // apply sigmoid
            if (debug==3) DEBUG_PRINT("3.FLOAT+SIG: Steering: %.3f, \tCollision: %.3f \n", (double)cnn_data_float[0], (double)cnn_data_float[1]);
            // Extract steering angle and proability of collision
            steering_angle = cnn_data_float[0];
            prob_of_col = cnn_data_float[1];

            // integrate the prob of collision
            if (enable_integral==1){
                prob_of_col = integrate_collision(prob_of_col, INTEGRAL_WEIGHT);
                if (debug==4) DEBUG_PRINT("4.INTEGRATED: Steering: %.3f, \tCollision (integ): %.3f \n", (double)steering_angle, (double)prob_integral);
            }

            // Calculate forward and anglar velocities
            compute_forward_velocity(prob_of_col, &forward_velocity);
            compute_angular_velocity(steering_angle, &angular_velocity);
            if (debug==5) DEBUG_PRINT("5.VELOCITIES: Angular: %.3f, \tForward: %.3f \n", (double)angular_velocity, (double)forward_velocity);

            // Low pass filtering of forward and angular_velocity
            if (enable_low_pass==1){
                forward_velocity = low_pass_filtering(forward_velocity, forward_velocity_old, alpha_vel);
                angular_velocity = low_pass_filtering(angular_velocity, angular_velocity_old, alpha_yaw);
                if (debug==6) DEBUG_PRINT("6.LOW_PASS: Angular: %.3f, \tForward: %.3f \n", (double)angular_velocity, (double)forward_velocity);
            }

            // emergency brake if prob_collision>threshold
            if (emergency_br==1){
                emergency_brake(&forward_velocity, CRITICAL_PROB_COLL);
                if (debug==7) DEBUG_PRINT("7.Emergency brake: Angular: %.3f, \tForward: %.3f \n", (double)angular_velocity, (double)forward_velocity);
            }

            if (debug==8) DEBUG_PRINT("8.FINAL: Angular: %.3f, \tForward: %.3f \n", (double)angular_velocity, (double)forward_velocity);

            // Give setpoint to the controller
            if (fly==1){
                setp_dronet = create_setpoint(forward_velocity, flying_height, angular_velocity);
                commanderSetSetpoint(&setp_dronet, 3);
            }

            // store i-1 values for the low pass filter
            forward_velocity_old = forward_velocity;
            angular_velocity_old = angular_velocity;
            // vTaskDelay(30);
        }
        if (dma_flag==0) // If no packet received for 400ms
        {
            timeout_counter++;
            if (timeout_counter > 10)
                DEBUG_PRINT("Navigation data timeout\n");
            vTaskDelay(40);
        }
    }
}

uint64_t t0, t_frame, t_prev;
// UART-DMA interrupt - triggered when a new inference result is available
void __attribute__((used)) DMA1_Stream1_IRQHandler(void)
{
    t_prev = t0;
    t0 = xTaskGetTickCount();
    t_frame = t0 - t_prev;
    DMA_ClearFlag(DMA1_Stream1, UART3_RX_DMA_ALL_FLAGS);
    dma_flag = 1;
}

/* --- TIP for Logging or parameters --- */
// The variable name: PARAM_ADD(TYPE, NAME, ADDRESS)
// both for logging (LOG_GROUP_START) or for parameters (PARAM_GROUP_START)
// should never exceed 9 CHARACTERS, otherwise the firmware won't start correctly

/* --- LOGGING --- */
LOG_GROUP_START(UART_LOG_GAP8)
LOG_ADD(LOG_FLOAT, gap8_steer, &cnn_data_float[0])
LOG_ADD(LOG_FLOAT, gap8_coll, &cnn_data_float[1])
LOG_GROUP_STOP(UART_LOG_GAP8)

LOG_GROUP_START(DRONET_LOG)
LOG_ADD(LOG_FLOAT, steering, &steering_angle)  	// CNN output
LOG_ADD(LOG_FLOAT, collision, &prob_of_col)		// CNN output after the sigmoid
LOG_ADD(LOG_FLOAT, fwd_vel, &forward_velocity)	// scaled and post-processed by low-pass, integrator, brake.
LOG_ADD(LOG_FLOAT, ang_vel, &angular_velocity)	// scaled and post-processed by low-pass, integrator, brake.
LOG_GROUP_STOP(DRONET_LOG)

/* --- PARAMETERS --- */
PARAM_GROUP_START(START_STOP)
PARAM_ADD(PARAM_UINT8, fly, &fly)
PARAM_GROUP_STOP(START_STOP)

// Activate - deactivate functionalities: 0=Non-active, 1=active
PARAM_GROUP_START(FUNCTIONALITIES)
PARAM_ADD(PARAM_UINT8, debug, &debug) 				// debug prints
PARAM_ADD(PARAM_UINT8, quadratic, &quadratic)		// activate quadratic preprocessing of probability of collision
PARAM_ADD(PARAM_UINT8, integrate, &enable_integral)	// integrate collision probability
PARAM_ADD(PARAM_UINT8, low_pass, &enable_low_pass)	// activate low pass filtering
PARAM_ADD(PARAM_UINT8, brake, &emergency_br)		// activate emergency brake if collision>threshold
PARAM_GROUP_STOP(FUNCTIONALITIES)

// Filters' parameters
PARAM_GROUP_START(DRONET_PARAMS)
PARAM_ADD(PARAM_FLOAT, fwd_index, &max_forward_index)
PARAM_ADD(PARAM_FLOAT, alpha_vel, &alpha_vel)
PARAM_ADD(PARAM_FLOAT, alpha_yaw, &alpha_yaw)
PARAM_ADD(PARAM_FLOAT, yaw_scale, &yaw_scaling)
PARAM_ADD(PARAM_FLOAT, height, &flying_height)
PARAM_ADD(PARAM_FLOAT, integ_w, &integral_weight)
PARAM_ADD(PARAM_FLOAT, integ_th, &integral_thresh)
PARAM_ADD(PARAM_FLOAT, brake_th, &critical_prob_collision)
PARAM_ADD(PARAM_FLOAT, quantum, &nemo_quantum)      // nemo quantum value
PARAM_GROUP_STOP(DRONET_PARAMS)