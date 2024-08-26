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

 File:    config_dronet.h
 Author:  Lorenzo Lamberti      <lorenzo.lamberti@unibo.it>
 Date:    01.03.2024
-------------------------------------------------------------------------------*/

/* --- SET TEST CONFIGURATION --- */
// #define COLLISION_TEST 1  // collision avoidance test with dynamic obstacle
//#define NAVIGATION_TEST 1  // navigation in corridors with turns
#define IOTJ_TEST_U 1  // specific config usef for IOTJ experiments


/* --- NEMO-DORY Flow --- */
#define NEMO_QUANTUM          0.0006  // NEMO-DORY quantization parameter

/* utils */
#define FINAL_LANDING_HEIGHT  0.07f   // [m] --> the drone drops at 0.07m of height


#ifdef COLLISION_TEST
    // FUNCTINOALITIES
    #define ENABLE_INTEGRAL     0
    #define ENABLE_LOW_PASS     1
    #define ENABLE_EMERGENCY_BR 1
    #define ENABLE_QUADRATIC    0

    // Scaling factors
    #define MAX_FORWARD_INDEX   0.5f   // Max forward speed [m/s].  Default: 1.0f
    #define YAW_SCALING         90.0f  // Max yaw-rate [yaw/s].  Default: 180.0f
    // Flight mission
    #define PULP_TARGET_H       0.50f  // Target height for drone's flight [m].  Default: 0.5f
    // Filters parameters
    #define ALPHA_VEL           0.3f   // Low pass filter parameter for forward speed. Range: [0.0<-weaker ; stronger->1.0]   Default: 0.5
    #define ALPHA_YAW           0.3f   // Low pass filter parameter for yaw rate.      Range: [0.0<-weaker ; stronger->1.0]   Default: 0.7
    // Integral parameters
    #define INTEGRAL_WEIGHT     0.2f   // Integral filter parameter for prob of collision. Default: 0.14f
    #define INTEGRAL_THRESH     0.6f   // Integral threshold for accumulating: prob_integral += (prob_of_col - thresh); Default=0.3f
    // Emergency brake
    #define CRITICAL_PROB_COLL  0.7f   // Default: 0.7f
#endif

#ifdef NAVIGATION_TEST
    // FUNCTINOALITIES
    #define ENABLE_INTEGRAL     0
    #define ENABLE_LOW_PASS     1
    #define ENABLE_EMERGENCY_BR 0
    #define ENABLE_QUADRATIC    0

    // Scaling factors
    #define MAX_FORWARD_INDEX   0.5f    // Max forward speed [m/s].  Default: 1.0f
    #define YAW_SCALING         120.0f  // Max yaw-rate [yaw/s].  Default: 180.0f
    // Flight mission
    #define PULP_TARGET_H       0.50f   // Target height for drone's flight [m].  Default: 0.5f
    // Filter parameters
    #define ALPHA_VEL           0.6f    // Low pass filter parameter for forward speed. Range: [0.0<-weaker ; stronger->1.0]   Default: 0.5
    #define ALPHA_YAW           0.6f    // Low pass filter parameter for yaw rate.      Range: [0.0<-weaker ; stronger->1.0]   Default: 0.7
    // Integral parameters
    #define INTEGRAL_WEIGHT     0.15f   // Integral filter parameter for prob of collision. Default: 0.14f
    #define INTEGRAL_THRESH     0.7f    // Integral threshold for accumulating: prob_integral += (prob_of_col - thresh); Default=0.3f
    // Emergency brake
    #define CRITICAL_PROB_COLL  0.7f    // Default: 0.7f
#endif

#ifdef IOTJ_TEST_U
    // FUNCTINOALITIES
    #define ENABLE_INTEGRAL     0
    #define ENABLE_LOW_PASS     1
    #define ENABLE_EMERGENCY_BR 0
    #define ENABLE_QUADRATIC    0

    // Scaling factors
    #define MAX_FORWARD_INDEX   0.5f    // Max forward speed [m/s].  Default: 1.0f
    #define YAW_SCALING         120.0f  // Max yaw-rate [yaw/s].  Default: 180.0f
    // Flight mission
    #define PULP_TARGET_H       0.50f   // Target height for drone's flight [m].  Default: 0.5f
    // Filter parameters
    #define ALPHA_VEL           0.3f    // Low pass filter parameter for forward speed. Range: [0.0<-weaker ; stronger->1.0]   Default: 0.5
    #define ALPHA_YAW           0.3f    // Low pass filter parameter for yaw rate.      Range: [0.0<-weaker ; stronger->1.0]   Default: 0.7
    // Integral parameters
    #define INTEGRAL_WEIGHT     0.15f   // Integral filter parameter for prob of collision. Default: 0.14f
    #define INTEGRAL_THRESH     0.7f    // Integral threshold for accumulating: prob_integral += (prob_of_col - thresh); Default=0.3f
    // Emergency brake
    #define CRITICAL_PROB_COLL  0.7f    // Default: 0.7f
#endif
