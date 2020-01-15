/*----------------------------------------------------------------------------*
 * Copyright (C) 2018-2019 ETH Zurich, Switzerland                            *
 * All rights reserved.                                                       *
 *                                                                            *
 * Licensed under the Apache License, Version 2.0 (the "License");            *
 * you may not use this file except in compliance with the License.           *
 * See LICENSE.apache.md in the top directory for details.                    *
 * You may obtain a copy of the License at                                    *
 *                                                                            *
 *     http://www.apache.org/licenses/LICENSE-2.0                             *
 *                                                                            *
 * Unless required by applicable law or agreed to in writing, software        *
 * distributed under the License is distributed on an "AS IS" BASIS,          *
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.   *
 * See the License for the specific language governing permissions and        *
 * limitations under the License.                                             *
 *                                                                            *
 * File:    PULPDronet.c                                                      *
 * Author:  Daniele Palossi <dpalossi@iis.ee.ethz.ch>                         *
 * Date:    10.04.2019                                                        *
 *----------------------------------------------------------------------------*/


#include <stdio.h>
#include "Gap8.h"
#include "PULPDronetKernels.h"
#include "config.h"
#include "Utils.h"


static char *			L2_base[NUM_L2_BUFF];
static char *			L2_next_free[NUM_L2_BUFF];
#ifdef PROFILE_CL
static int				perf_exe_cum_cl[NLAYERS];
static int				perf_mem_cum_cl[NLAYERS];
#endif
static rt_hyperram_t *	hyperram;
static rt_camera_t *	camera;
static int				imgTransferDone = 0;
static rt_event_t *		event_capture;
static rt_event_t *		event_cluster;
static int 				outputSizesB[NLAYERS];
static short int *		L2_image;
static short int *		L2_bias[NWEIGTHS];
static short int *		L3_weights[NWEIGTHS];
static int				L3_sizes[NWEIGTHS]; 
static unsigned int		L2_bias_sizes[NWEIGTHS]; 
static int 				Norm_Factor[NWEIGTHS];
static short int		SPIM_tx[2];
static short int		SPIM_rx[2];


unsigned int PMU_set_voltage(unsigned int Voltage, unsigned int CheckFrequencies);


static void handle_error() {
	int error = rt_error_current();
	printf("Caught error (error code 0x%x): %s\n", error, rt_error_str(error));
	exit(-1);
}


static void handle_async_error(void *arg, rt_event_t *event, int error, void *object) {
	printf("Received error (error code 0x%x, event %p, object: %p): %s\n", error, event, object, rt_error_str(error));
	exit(-1);
}


static void enqueue_capture();


static char * meta_alloc(int memId, int sizeBytes) {

	char * return_ptr = L2_next_free[memId];
	if(L2_next_free[memId] + sizeBytes > L2_base[memId] + L2_buffers_size[memId]) {
		printf("META ALLOC ERROR on MEM %d. Request: %d Byte Available: %d Byte\n", 
			memId, 
			sizeBytes, 
			L2_buffers_size[memId]-(int)((unsigned int)L2_next_free[memId]-(unsigned int)L2_base[memId]));
		exit(-1);
	}
	L2_next_free[memId] += sizeBytes;
	return return_ptr;
}


static void meta_free(int memId, int sizeBytes) {

	if(L2_next_free[memId] - sizeBytes < L2_base[memId]) {
		printf("META FREE ERROR on MEM %d. Request: %d Byte\n", memId, sizeBytes);
		exit(-1);
	}
	L2_next_free[memId] -= sizeBytes;
}


static void L3toL2(short int *bufferL3, short int *L2_weights, int sizeL3B) {

	rt_hyperram_req_t reqL3;
	rt_hyperram_cluster_read(hyperram, L2_weights, bufferL3, sizeL3B, &reqL3); 
	rt_hyperram_cluster_wait(&reqL3);
}


#ifdef CHECKSUM
static void check_layer(short int *output, int layer) {

	unsigned int checksum = 0;
	short unsigned int *ptr = (short unsigned int *) output;
	for(int j=0; j<outCh[layer]*outW[layer]*outH[layer]; j++) {
		checksum += (unsigned int) ptr[j];
	}

	if(Layer_GT[layer] == checksum)
		printf("Checksum Layer %d:\tOk\n", layer+1);
	else 
		printf("Checksum Layer %d:\tFailed [%u vs. %u]\n", layer+1, checksum, Layer_GT[layer]);
}
#endif


static void end_of_frame() {

	rt_cam_control(camera, CMD_PAUSE, NULL);

#if CROPPING == 1
	unsigned char * origin 		= (unsigned char *) L2_image;
	unsigned char * ptr_crop 	= (unsigned char *) L2_image;
	int 			init_offset = CAM_FULLRES_W * LL_Y + LL_X; 
	int 			outid 		= 0;
	
	for(int i=0; i<CAM_CROP_H; i++) {	
		rt_event_execute(NULL, 0);
		unsigned char * line = ptr_crop + init_offset + CAM_FULLRES_W * i;
		for(int j=0; j<CAM_CROP_W; j++) {
			origin[outid] = line[j];
			outid++;
		}
	}
#endif

	unsigned char * ptr = (unsigned char *) L2_image;

	for(int i=CAM_CROP_H-1; i>=0; i--) {
		rt_event_execute(NULL, 0);
		for(int j=CAM_CROP_W-1; j>=0; j--) {
			L2_image[i*CAM_CROP_W+j] = (short int) ptr[i*CAM_CROP_W+j];
		}
	}

	imgTransferDone = 1;
}


static void enqueue_capture() {

	rt_cam_control(camera, CMD_START, NULL);

	rt_camera_capture(camera, (unsigned char*)L2_image, CAM_WIDTH*CAM_HEIGHT*sizeof(unsigned char), rt_event_get(NULL, end_of_frame, NULL));
}


/*----------------.  .----------------.  .-----------------.
| .--------------. || .--------------. || .--------------. |
| |  _______     | || | _____  _____ | || | ____  _____  | |
| | |_   __ \    | || ||_   _||_   _|| || ||_   \|_   _| | |
| |   | |__) |   | || |  | |    | |  | || |  |   \ | |   | |
| |   |  __ /    | || |  | '    ' |  | || |  | |\ \| |   | |
| |  _| |  \ \_  | || |   \ `--' /   | || | _| |_\   |_  | |
| | |____| |___| | || |    `.__.'    | || ||_____|\____| | |
| |              | || |              | || |              | |
| '--------------' || '--------------' || '--------------' |
 '----------------'  '----------------'  '----------------*/

static void RunPULPDronet() {

	short int *L2_input;
	short int *L2_weights;
	short int *L2_output[NLAYERS];
	int memId_O = 0;
	int memId_I = 0;
	int memId_W = 0;

#ifdef PROFILE_CL
	int perf_start;
	rt_perf_t perf_cl;
	rt_perf_init(&perf_cl);
	rt_perf_conf(&perf_cl, (1<<RT_PERF_CYCLES));
	rt_perf_reset(&perf_cl);
	rt_perf_start(&perf_cl);
#endif


/* --------------------------------- LAYER 1 -------------------------------- */
	L2_input = L2_image;
	memId_O = 0;
	memId_W = 0;

#if defined(DUMP_I) && (DUMP_I==0 || DUMP_I==18)
	dumpFMs(0, L2_input, 0, DUMP_T);
#endif

#ifdef PROFILE_CL
	perf_start = rt_perf_read(RT_PERF_CYCLES);
#endif

	L2_output[0] = (short int *) meta_alloc(memId_O, outputSizesB[0]);

	L2_weights = (short int *) meta_alloc(memId_W, L3_sizes[0]);

	L3toL2(L3_weights[0], L2_weights, L3_sizes[0]);

#ifdef PROFILE_CL
	perf_mem_cum_cl[0] = rt_perf_read(RT_PERF_CYCLES) - perf_start;
	perf_start = rt_perf_read(RT_PERF_CYCLES);
#endif

	LargeParConv_5x5_S2_Max2x2_S2_H_1(L2_input, L2_weights, L2_output[0], Norm_Factor[0], L2_bias[0], 0);

#ifdef PROFILE_CL
	perf_exe_cum_cl[0] = rt_perf_read(RT_PERF_CYCLES) - perf_start;
#endif

#if defined(DUMP_W) && (DUMP_W==0 || DUMP_W==18)
	dumpW(0, L2_weights, DUMP_T);
#endif

#if defined(DUMP_B) && (DUMP_B==0 || DUMP_B==18)
	dumpBias(0, L2_bias[0], DUMP_T);
#endif

#if defined(DUMP_O) && (DUMP_O==0 || DUMP_O==18) 
	dumpFMs(0, L2_output[0], 1, DUMP_T);
#endif

#ifdef CHECKSUM
	check_layer(L2_output[0], 0);
#endif

	meta_free(memId_W, L3_sizes[0]);


/* -------------------- TRIGGER A NEW IMG TRANSFER ON FC -------------------- */

__rt_cluster_push_fc_event(event_capture);

/* --------------------------------- LAYER 2 -------------------------------- */
	L2_input = L2_output[0];
	memId_O = 0;

#if defined(DUMP_I) && (DUMP_I==1 || DUMP_I==18)
	dumpFMs(1, L2_input, 0, DUMP_T);
#endif

#ifdef PROFILE_CL
	perf_mem_cum_cl[1] = 0;
	perf_start = rt_perf_read(RT_PERF_CYCLES);
#endif

	L2_output[1] = (short int *) meta_alloc(memId_O, outputSizesB[1]);

	ReLU_SW_1(L2_input, L2_output[1], 0);

#ifdef PROFILE_CL
	perf_exe_cum_cl[1] = rt_perf_read(RT_PERF_CYCLES) - perf_start;
#endif

#if defined(DUMP_O) && (DUMP_O==1 || DUMP_O==18)
	dumpFMs(1, L2_output[1], 1, DUMP_T);
#endif

#ifdef CHECKSUM
	check_layer(L2_output[1], 1);
#endif


	L2_input = L2_output[1];
	memId_O = 1;
	memId_W = 1;

#if defined(DUMP_I) && (DUMP_I==2 || DUMP_I==18)
	dumpFMs(2, L2_input, 0, DUMP_T);
#endif

#ifdef PROFILE_CL
	perf_start = rt_perf_read(RT_PERF_CYCLES);
#endif

	L2_output[2] = (short int *) meta_alloc(memId_O, outputSizesB[2]);
	L2_weights = (short int *) meta_alloc(memId_W, L3_sizes[1]);

	L3toL2(L3_weights[1], L2_weights, L3_sizes[1]);

#ifdef PROFILE_CL
	perf_mem_cum_cl[2] = rt_perf_read(RT_PERF_CYCLES) - perf_start;
	perf_start = rt_perf_read(RT_PERF_CYCLES);
#endif

	MedParConv_3x3_S2_ReLU_2(L2_input, L2_weights, L2_output[2], Norm_Factor[1], L2_bias[1], 0);

#ifdef PROFILE_CL
	perf_exe_cum_cl[2] = rt_perf_read(RT_PERF_CYCLES) - perf_start;
#endif

#if defined(DUMP_W) && (DUMP_W==2 || DUMP_W==18)
	dumpW(2, L2_weights, DUMP_T);
#endif

#if defined(DUMP_B) && (DUMP_B==2 || DUMP_B==18)
	dumpBias(2, L2_bias[1], DUMP_T);
#endif

#if defined(DUMP_O) && (DUMP_O==2 || DUMP_O==18)
	dumpFMs(2, L2_output[2], 1, DUMP_T);
#endif

#ifdef CHECKSUM
	check_layer(L2_output[2], 2);
#endif

	meta_free(memId_W, L3_sizes[1]);
	meta_free(0, outputSizesB[1]);


/* --------------------------------- LAYER 3 -------------------------------- */
	L2_input = L2_output[2];
	memId_O = 0;
	memId_W = 1;

#if defined(DUMP_I) && (DUMP_I==3 || DUMP_I==18)
	dumpFMs(3, L2_input, 0, DUMP_T);
#endif

#ifdef PROFILE_CL
	perf_start = rt_perf_read(RT_PERF_CYCLES);
#endif

	L2_output[3] = (short int *) meta_alloc(memId_O, outputSizesB[3]);
	L2_weights = (short int *) meta_alloc(memId_W, L3_sizes[2]);

	L3toL2(L3_weights[2], L2_weights, L3_sizes[2]);

#ifdef PROFILE_CL
	perf_mem_cum_cl[3] = rt_perf_read(RT_PERF_CYCLES) - perf_start;
	perf_start = rt_perf_read(RT_PERF_CYCLES);
#endif

	MedParConv_3x3_S1_3(L2_input, L2_weights, L2_output[3], Norm_Factor[2], L2_bias[2], 0);

#ifdef PROFILE_CL
	perf_exe_cum_cl[3] = rt_perf_read(RT_PERF_CYCLES) - perf_start;
#endif

#if defined(DUMP_W) && (DUMP_W==3 || DUMP_W==18)
	dumpW(3, L2_weights, DUMP_T);
#endif

#if defined(DUMP_B) && (DUMP_B==3 || DUMP_B==18)
	dumpBias(3, L2_bias[2], DUMP_T);
#endif

#if defined(DUMP_O) && (DUMP_O==3 || DUMP_O==18)
	dumpFMs(3, L2_output[3], 1, DUMP_T);
#endif

#ifdef CHECKSUM
	check_layer(L2_output[3], 3);
#endif

	meta_free(memId_W, L3_sizes[2]);
	meta_free(1, outputSizesB[2]);
	

/* --------------------------------- LAYER 4 -------------------------------- */
	L2_input = L2_output[0];
	memId_O = 1;
	memId_W = 1;

#if defined(DUMP_I) && (DUMP_I==4 || DUMP_I==18)
	dumpFMs(4, L2_input, 0, DUMP_T);
#endif

#ifdef PROFILE_CL
	perf_start = rt_perf_read(RT_PERF_CYCLES);
#endif

	L2_output[4] = (short int *) meta_alloc(memId_O, outputSizesB[4]);
	L2_weights = (short int *) meta_alloc(memId_W, L3_sizes[3]);

	L3toL2(L3_weights[3], L2_weights, L3_sizes[3]);

#ifdef PROFILE_CL
	perf_mem_cum_cl[4] = rt_perf_read(RT_PERF_CYCLES) - perf_start;
	perf_start = rt_perf_read(RT_PERF_CYCLES);
#endif

	MedParConv_1x1_S2_4(L2_input, L2_weights, L2_output[4], Norm_Factor[3], L2_bias[3], 0);

#ifdef PROFILE_CL
	perf_exe_cum_cl[4] = rt_perf_read(RT_PERF_CYCLES) - perf_start;
#endif

#if defined(DUMP_W) && (DUMP_W==4 || DUMP_W==18)
	dumpW(4, L2_weights, DUMP_T);
#endif

#if defined(DUMP_B) && (DUMP_B==4 || DUMP_B==18)
	dumpBias(4, L2_bias[3], DUMP_T);
#endif

#if defined(DUMP_O) && (DUMP_O==4 || DUMP_O==18)
	dumpFMs(4, L2_output[4], 1, DUMP_T);
#endif

#ifdef CHECKSUM
	check_layer(L2_output[4], 4);
#endif

	meta_free(memId_W, L3_sizes[3]);
	

/* -------------------------------- ADD RES 1 -------------------------------- */
	L2_input = L2_output[3];
	L2_output[5] = L2_output[4];

#if defined(DUMP_I) && (DUMP_I==5 || DUMP_I==18)
	dumpFMs(5, L2_input, 0, DUMP_T);
#endif

#ifdef PROFILE_CL
	perf_mem_cum_cl[5] = 0;
	perf_start = rt_perf_read(RT_PERF_CYCLES);
#endif

	AddFeatureMaps_SW_1(L2_input, L2_output[5], 0);

#ifdef PROFILE_CL
	perf_exe_cum_cl[5] = rt_perf_read(RT_PERF_CYCLES) - perf_start;
#endif

#if defined(DUMP_O) && (DUMP_O==5 || DUMP_O==18)
	dumpFMs(5, L2_output[5], 1, DUMP_T);
#endif

#ifdef CHECKSUM
	check_layer(L2_output[5], 5);
#endif

	meta_free(0, outputSizesB[3]);
	meta_free(0, outputSizesB[0]);
	

/* --------------------------------- LAYER 5 -------------------------------- */
	L2_input = L2_output[5];
	memId_O = 0;

#if defined(DUMP_I) && (DUMP_I==6 || DUMP_I==18)
	dumpFMs(6, L2_input, 0, DUMP_T);
#endif

#ifdef PROFILE_CL
	perf_mem_cum_cl[6] = 0;
	perf_start = rt_perf_read(RT_PERF_CYCLES);
#endif

	L2_output[6] = (short int *) meta_alloc(memId_O, outputSizesB[6]);

	ReLU_SW_2(L2_input, L2_output[6], 0);

#ifdef PROFILE_CL
	perf_exe_cum_cl[6] = rt_perf_read(RT_PERF_CYCLES) - perf_start;
#endif

#if defined(DUMP_O) && (DUMP_O==6 || DUMP_O==18)
	dumpFMs(6, L2_output[6], 1, DUMP_T);
#endif

#ifdef CHECKSUM
	check_layer(L2_output[6], 6);
#endif
	

	L2_input = L2_output[6];
	memId_O = 0;
	memId_W = 0;

#if defined(DUMP_I) && (DUMP_I==7 || DUMP_I==18)
	dumpFMs(7, L2_input, 0, DUMP_T);
#endif

#ifdef PROFILE_CL
	perf_start = rt_perf_read(RT_PERF_CYCLES);
#endif

	L2_output[7] = (short int *) meta_alloc(memId_O, outputSizesB[7]);
	L2_weights = (short int *) meta_alloc(memId_W, L3_sizes[4]);

	L3toL2(L3_weights[4], L2_weights, L3_sizes[4]);

#ifdef PROFILE_CL
	perf_mem_cum_cl[7] = rt_perf_read(RT_PERF_CYCLES) - perf_start;
	perf_start = rt_perf_read(RT_PERF_CYCLES);
#endif

	MedParConv_3x3_S2_ReLU_5(L2_input, L2_weights, L2_output[7], Norm_Factor[4], L2_bias[4], 0);

#ifdef PROFILE_CL
	perf_exe_cum_cl[7] = rt_perf_read(RT_PERF_CYCLES) - perf_start;
#endif

#if defined(DUMP_W) && (DUMP_W==7 || DUMP_W==18)
	dumpW(7, L2_weights, DUMP_T);
#endif

#if defined(DUMP_B) && (DUMP_B==7 || DUMP_B==18)
	dumpBias(7, L2_bias[4], DUMP_T);
#endif

#if defined(DUMP_O) && (DUMP_O==7 || DUMP_O==18)
	dumpFMs(7, L2_output[7], 1, DUMP_T);
#endif

#ifdef CHECKSUM
	check_layer(L2_output[7], 7);
#endif

	meta_free(memId_W, L3_sizes[4]);
	

/* --------------------------------- LAYER 6 -------------------------------- */
	L2_input = L2_output[7];
	memId_O = 1;
	memId_W = 0;

#if defined(DUMP_I) && (DUMP_I==8 || DUMP_I==18)
	dumpFMs(8, L2_input, 0, DUMP_T);
#endif

#ifdef PROFILE_CL
	perf_start = rt_perf_read(RT_PERF_CYCLES);
#endif

	L2_output[8] = (short int *) meta_alloc(memId_O, outputSizesB[8]);
	L2_weights = (short int *) meta_alloc(memId_W, L3_sizes[5]);

	L3toL2(L3_weights[5], L2_weights, L3_sizes[5]);

#ifdef PROFILE_CL
	perf_mem_cum_cl[8] = rt_perf_read(RT_PERF_CYCLES) - perf_start;
	perf_start = rt_perf_read(RT_PERF_CYCLES);
#endif

	MedParConv_3x3_S1_6(L2_input, L2_weights, L2_output[8], Norm_Factor[5], L2_bias[5], 0);

#ifdef PROFILE_CL
	perf_exe_cum_cl[8] = rt_perf_read(RT_PERF_CYCLES) - perf_start;
#endif

#if defined(DUMP_W) && (DUMP_W==8 || DUMP_W==18)
	dumpW(8, L2_weights, DUMP_T);
#endif

#if defined(DUMP_B) && (DUMP_B==8 || DUMP_B==18)
	dumpBias(8, L2_bias[5], DUMP_T);
#endif

#if defined(DUMP_O) && (DUMP_O==8 || DUMP_O==18)
	dumpFMs(8, L2_output[8], 1, DUMP_T);
#endif

#ifdef CHECKSUM
	check_layer(L2_output[8], 8);
#endif

	meta_free(memId_W, L3_sizes[5]);
	meta_free(0, outputSizesB[7]);
	meta_free(0, outputSizesB[6]);
	

/* --------------------------------- LAYER 7 -------------------------------- */
	L2_input = L2_output[4];
	memId_O = 0;
	memId_W = 0;

#if defined(DUMP_I) && (DUMP_I==9 || DUMP_I==18)
	dumpFMs(9, L2_input, 0, DUMP_T);
#endif

#ifdef PROFILE_CL
	perf_start = rt_perf_read(RT_PERF_CYCLES);
#endif

	L2_output[9] = (short int *) meta_alloc(memId_O, outputSizesB[9]);
	L2_weights = (short int *) meta_alloc(memId_W, L3_sizes[6]);

	L3toL2(L3_weights[6], L2_weights, L3_sizes[6]);

#ifdef PROFILE_CL
	perf_mem_cum_cl[9] = rt_perf_read(RT_PERF_CYCLES) - perf_start;
	perf_start = rt_perf_read(RT_PERF_CYCLES);
#endif

	MedParConv_1x1_S2_7(L2_input, L2_weights, L2_output[9], Norm_Factor[6], L2_bias[6], 0);

#ifdef PROFILE_CL
	perf_exe_cum_cl[9] = rt_perf_read(RT_PERF_CYCLES) - perf_start;
#endif

#if defined(DUMP_W) && (DUMP_W==9 || DUMP_W==18)
	dumpW(9, L2_weights, DUMP_T);
#endif

#if defined(DUMP_B) && (DUMP_B==9 || DUMP_B==18)
	dumpBias(9, L2_bias[6], DUMP_T);
#endif

#if defined(DUMP_O) && (DUMP_O==9 || DUMP_O==18)
	dumpFMs(9, L2_output[9], 1, DUMP_T);
#endif

#ifdef CHECKSUM
	check_layer(L2_output[9], 9);
#endif

	meta_free(memId_W, L3_sizes[6]);
	

/* -------------------------------- ADD RES 2 -------------------------------- */
	L2_input = L2_output[8];
	L2_output[10] = L2_output[9];

#if defined(DUMP_I) && (DUMP_I==10 || DUMP_I==18)
	dumpFMs(10, L2_input, 0, DUMP_T);
#endif

#ifdef PROFILE_CL
	perf_mem_cum_cl[10] = 0;
	perf_start = rt_perf_read(RT_PERF_CYCLES);
#endif

	AddFeatureMaps_SW_2(L2_input, L2_output[10], 0);

#ifdef PROFILE_CL
	perf_exe_cum_cl[10] = rt_perf_read(RT_PERF_CYCLES) - perf_start;
#endif

#if defined(DUMP_O) && (DUMP_O==10 || DUMP_O==18)
	dumpFMs(10, L2_output[10], 1, DUMP_T);
#endif

#ifdef CHECKSUM
	check_layer(L2_output[10], 10);
#endif

	meta_free(1, outputSizesB[8]);
	meta_free(1, outputSizesB[4]);
	

/* --------------------------------- LAYER 8 -------------------------------- */
	L2_input = L2_output[10];
	memId_O = 0;

#if defined(DUMP_I) && (DUMP_I==11 || DUMP_I==18)
	dumpFMs(11, L2_input, 0, DUMP_T);
#endif

#ifdef PROFILE_CL
	perf_mem_cum_cl[11] = 0;
	perf_start = rt_perf_read(RT_PERF_CYCLES);
#endif

	L2_output[11] = (short int *) meta_alloc(memId_O, outputSizesB[11]);

	ReLU_SW_3(L2_input, L2_output[11], 0);

#ifdef PROFILE_CL
	perf_exe_cum_cl[11] = rt_perf_read(RT_PERF_CYCLES) - perf_start;
#endif

#if defined(DUMP_O) && (DUMP_O==11 || DUMP_O==18)
	dumpFMs(11, L2_output[11], 1, DUMP_T);
#endif

#ifdef CHECKSUM
	check_layer(L2_output[11], 11);
#endif
	

	L2_input = L2_output[11];
	memId_O = 1;
	memId_W = 0;

#if defined(DUMP_I) && (DUMP_I==12 || DUMP_I==18)
	dumpFMs(12, L2_input, 0, DUMP_T);
#endif

#ifdef PROFILE_CL
	perf_start = rt_perf_read(RT_PERF_CYCLES);
#endif

	L2_output[12] = (short int *) meta_alloc(memId_O, outputSizesB[12]);
	L2_weights = (short int *) meta_alloc(memId_W, L3_sizes[7]);

	L3toL2(L3_weights[7], L2_weights, L3_sizes[7]);

#ifdef PROFILE_CL
	perf_mem_cum_cl[12] = rt_perf_read(RT_PERF_CYCLES) - perf_start;
	perf_start = rt_perf_read(RT_PERF_CYCLES);
#endif

	MedParConv_3x3_S2_ReLU_8(L2_input, L2_weights, L2_output[12], Norm_Factor[7], L2_bias[7], 0);

#ifdef PROFILE_CL
	perf_exe_cum_cl[12] = rt_perf_read(RT_PERF_CYCLES) - perf_start;
#endif

#if defined(DUMP_W) && (DUMP_W==12 || DUMP_W==18)
	dumpW(12, L2_weights, DUMP_T);
#endif

#if defined(DUMP_B) && (DUMP_B==12 || DUMP_B==18)
	dumpBias(12, L2_bias[7], DUMP_T);
#endif

#if defined(DUMP_O) && (DUMP_O==12 || DUMP_O==18)
	dumpFMs(12, L2_output[12], 1, DUMP_T);
#endif

#ifdef CHECKSUM
	check_layer(L2_output[12], 12);
#endif

	meta_free(memId_W, L3_sizes[7]);
	meta_free(0, outputSizesB[11]);
	

/* --------------------------------- LAYER 9 -------------------------------- */
	L2_input = L2_output[12];
	memId_O = 1;
	memId_W = 0;

#if defined(DUMP_I) && (DUMP_I==13 || DUMP_I==18)
	dumpFMs(13, L2_input, 0, DUMP_T);
#endif

#ifdef PROFILE_CL
	perf_start = rt_perf_read(RT_PERF_CYCLES);
#endif

	L2_output[13] = (short int *) meta_alloc(memId_O, outputSizesB[13]);
	L2_weights = (short int *) meta_alloc(memId_W, L3_sizes[8]);

	L3toL2(L3_weights[8], L2_weights, L3_sizes[8]);

#ifdef PROFILE_CL
	perf_mem_cum_cl[13] = rt_perf_read(RT_PERF_CYCLES) - perf_start;
	perf_start = rt_perf_read(RT_PERF_CYCLES);
#endif

	MedParConv_3x3_S1_9(L2_input, L2_weights, L2_output[13], Norm_Factor[8], L2_bias[8], 0);

#ifdef PROFILE_CL
	perf_exe_cum_cl[13] = rt_perf_read(RT_PERF_CYCLES) - perf_start;
#endif

#if defined(DUMP_W) && (DUMP_W==13 || DUMP_W==18)
	dumpW(13, L2_weights, DUMP_T);
#endif

#if defined(DUMP_B) && (DUMP_B==13 || DUMP_B==18)
	dumpBias(13, L2_bias[8], DUMP_T);
#endif

#if defined(DUMP_O) && (DUMP_O==13 || DUMP_O==18)
	dumpFMs(13, L2_output[13], 1, DUMP_T);
#endif

#ifdef CHECKSUM
	check_layer(L2_output[13], 13);
#endif

	meta_free(memId_W, L3_sizes[8]);
	

/* --------------------------------- LAYER 10 ------------------------------- */
	L2_input = L2_output[9];
	memId_O = 0;
	memId_W = 1;

#if defined(DUMP_I) && (DUMP_I==14 || DUMP_I==18)
	dumpFMs(14, L2_input, 0, DUMP_T);
#endif

#ifdef PROFILE_CL
	perf_start = rt_perf_read(RT_PERF_CYCLES);
#endif

	L2_output[14] = (short int *) meta_alloc(memId_O, outputSizesB[14]);
	L2_weights = (short int *) meta_alloc(memId_W, L3_sizes[9]);

	L3toL2(L3_weights[9], L2_weights, L3_sizes[9]);

#ifdef PROFILE_CL
	perf_mem_cum_cl[14] = rt_perf_read(RT_PERF_CYCLES) - perf_start;
	perf_start = rt_perf_read(RT_PERF_CYCLES);
#endif

	MedParConv_1x1_S1_ReLU_10(L2_input, L2_weights, L2_output[14], Norm_Factor[9], L2_bias[9], 0);

#ifdef PROFILE_CL
	perf_exe_cum_cl[14] = rt_perf_read(RT_PERF_CYCLES) - perf_start;
#endif

#if defined(DUMP_W) && (DUMP_W==14 || DUMP_W==18)
	dumpW(14, L2_weights, DUMP_T);
#endif

#if defined(DUMP_B) && (DUMP_B==14 || DUMP_B==18)
	dumpBias(14, L2_bias[9], DUMP_T);
#endif

#if defined(DUMP_O) && (DUMP_O==14 || DUMP_O==18)
	dumpFMs(14, L2_output[14], 1, DUMP_T);
#endif

#ifdef CHECKSUM
	check_layer(L2_output[14], 14);
#endif

	meta_free(memId_W, L3_sizes[9]);


/* -------------------------------- ADD RES 3 -------------------------------- */
	L2_input = L2_output[13];
	L2_output[15] = L2_output[14];

#if defined(DUMP_I) && (DUMP_I==15 || DUMP_I==18)
	dumpFMs(15, L2_input, 0, DUMP_T);
#endif

#ifdef PROFILE_CL
	perf_mem_cum_cl[15] = 0;
	perf_start = rt_perf_read(RT_PERF_CYCLES);
#endif

	AddFeatureMapsReLu_SW_3(L2_input, L2_output[15], 0);

#ifdef PROFILE_CL
	perf_exe_cum_cl[15] = rt_perf_read(RT_PERF_CYCLES) - perf_start;
#endif

#if defined(DUMP_O) && (DUMP_O==15 || DUMP_O==18)
	dumpFMs(15, L2_output[15], 1, DUMP_T);
#endif

#ifdef CHECKSUM
	check_layer(L2_output[15], 15);
#endif

	meta_free(1, outputSizesB[13]);
	meta_free(1, outputSizesB[12]);


/* --------------------------------- DENSE 1 -------------------------------- */
	L2_input = L2_output[15];
	memId_O = 1;
	memId_W = 1;

#if defined(DUMP_I) && (DUMP_I==16 || DUMP_I==18)
	dumpFMs(16, L2_input, 0, DUMP_T);
#endif

#ifdef PROFILE_CL
	perf_start = rt_perf_read(RT_PERF_CYCLES);
#endif

	L2_output[16] = (short int *) meta_alloc(memId_O, outputSizesB[16]+2);
	L2_weights = (short int *) meta_alloc(memId_W, L3_sizes[10]);

	L3toL2(L3_weights[10], L2_weights, L3_sizes[10]);

#ifdef PROFILE_CL
	perf_mem_cum_cl[16] = rt_perf_read(RT_PERF_CYCLES) - perf_start;
	perf_start = rt_perf_read(RT_PERF_CYCLES);
#endif

	LinearLayer_SW_1(L2_input, L2_weights, Norm_Factor[10], L2_bias[10], NORM_BIAS_DENSE, L2_output[16], 0, 0);

#ifdef PROFILE_CL
	perf_exe_cum_cl[16] = rt_perf_read(RT_PERF_CYCLES) - perf_start;
#endif

#if defined(DUMP_W) && (DUMP_W==16 || DUMP_W==18)
	dumpW(16, L2_weights, DUMP_T);
#endif

#if defined(DUMP_B) && (DUMP_B==16 || DUMP_B==18)
	dumpBias(16, L2_bias[10], DUMP_T);
#endif

#if defined(DUMP_O) && (DUMP_O==16 || DUMP_O==18)
	dumpFMs(16, L2_output[16], 1, DUMP_T);
#endif

#ifdef CHECKSUM
	check_layer(L2_output[16], 16);
#endif

	meta_free(memId_W, L3_sizes[10]);
	SPIM_tx[0] = L2_output[16][0];
	meta_free(memId_O, outputSizesB[16]+2);


/* --------------------------------- DENSE 2 -------------------------------- */
	L2_input = L2_output[15];
	memId_O = 1;
	memId_W = 1;

#if defined(DUMP_I) && (DUMP_I==17 || DUMP_I==18)
	dumpFMs(17, L2_input, 0, DUMP_T);
#endif

#ifdef PROFILE_CL
	perf_start = rt_perf_read(RT_PERF_CYCLES);
#endif

	L2_output[17] = (short int *) meta_alloc(memId_O, outputSizesB[17]+2);
	L2_weights = (short int *) meta_alloc(memId_W, L3_sizes[11]);

	L3toL2(L3_weights[11], L2_weights, L3_sizes[11]);

#ifdef PROFILE_CL
	perf_mem_cum_cl[17] = rt_perf_read(RT_PERF_CYCLES) - perf_start;
	perf_start = rt_perf_read(RT_PERF_CYCLES);
#endif

	LinearLayer_SW_2(L2_input, L2_weights, Norm_Factor[11], L2_bias[11], NORM_BIAS_DENSE, L2_output[17], 0, 0);

#ifdef PROFILE_CL
	perf_exe_cum_cl[17] = rt_perf_read(RT_PERF_CYCLES) - perf_start;
#endif

#if defined(DUMP_W) && (DUMP_W==17 || DUMP_W==18)
	dumpW(17, L2_weights, DUMP_T);
#endif

#if defined(DUMP_B) && (DUMP_B==17 || DUMP_B==18)
	dumpBias(17, L2_bias[11], DUMP_T);
#endif

#if defined(DUMP_O) && (DUMP_O==17 || DUMP_O==18)
	dumpFMs(17, L2_output[17], 1, DUMP_T);
#endif

#ifdef CHECKSUM
	check_layer(L2_output[17], 17);
#endif

	meta_free(memId_W, L3_sizes[11]);
	SPIM_tx[1] = L2_output[17][0];
	meta_free(memId_O, outputSizesB[17]+2);
	meta_free(0, outputSizesB[14]);
	meta_free(0, outputSizesB[9]);


/* -------------------------------------------------------------------------- */


#ifdef PROFILE_CL
	rt_perf_stop(&perf_cl);
	rt_perf_save(&perf_cl);

	printf("CL Cycles:\t\tL3/L2 Memory\t\tExecution\n");
	for(int i=0; i<NLAYERS; i++) {
		printf("Layer %2d\t\t%10d\t\t%10d\n", i+1, perf_mem_cum_cl[i], perf_exe_cum_cl[i]);
	}
	printf("Total (mem+exe): %10d\n", rt_perf_get(&perf_cl, RT_PERF_CYCLES));
#endif

}


/*----------------.  .----------------.  .----------------.  .-----------------.
| .--------------. || .--------------. || .--------------. || .--------------. |
| | ____    ____ | || |      __      | || |     _____    | || | ____  _____  | |
| ||_   \  /   _|| || |     /  \     | || |    |_   _|   | || ||_   \|_   _| | |
| |  |   \/   |  | || |    / /\ \    | || |      | |     | || |  |   \ | |   | |
| |  | |\  /| |  | || |   / ____ \   | || |      | |     | || |  | |\ \| |   | |
| | _| |_\/_| |_ | || | _/ /    \ \_ | || |     _| |_    | || | _| |_\   |_  | |
| ||_____||_____|| || ||____|  |____|| || |    |_____|   | || ||_____|\____| | |
| |              | || |              | || |              | || |              | |
| '--------------' || '--------------' || '--------------' || '--------------' |
 '----------------'  '----------------'  '----------------'  '----------------*/ 

int main() {

#ifdef VERBOSE
	printf("FC Launched\n");
#endif

/* ----------------------------- SET FREQUENCIES ---------------------------- */

		PMU_set_voltage(VOLTAGE_SOC, 0);
		rt_time_wait_us(10000);
		rt_freq_set(RT_FREQ_DOMAIN_FC, FREQUENCY_FC);
		rt_time_wait_us(10000);
		rt_freq_set(RT_FREQ_DOMAIN_CL, FREQUENCY_CL);
		rt_time_wait_us(10000);

/* -------------------------------------------------------------------------- */


/* ----------------------------- PADFRAME SETUP ----------------------------- */

	rt_padframe_profile_t *profile_hyper = rt_pad_profile_get("hyper");
#ifdef VERBOSE
	printf("Pads configuration:\t\t\t%s\n", profile_hyper?"Ok":"Failed");
#endif
	if(profile_hyper == NULL) return -1;
	rt_padframe_set(profile_hyper);

/* -------------------------------------------------------------------------- */


/* --------------------------------- LOAD L3 -------------------------------- */

	/* ----------------- HyperRAM ----------------- */
	rt_hyperram_conf_t hyperram_conf;
	rt_hyperram_conf_init(&hyperram_conf);

	hyperram_conf.id = 0;
	hyperram_conf.ram_size = 8<<20;

	hyperram = rt_hyperram_open(NULL, &hyperram_conf, NULL);
	if(hyperram == NULL) {
#ifdef VERBOSE
		printf("Error Opening HyperRAM\n");
#endif
		return -1;
	}

	/* ---------------- HyperFlash ---------------- */
	rt_fs_t *fs;
	rt_file_t *file;
	rt_fs_conf_t hyperflash_conf;
	rt_error_conf(NULL, handle_async_error, NULL);
	if(rt_event_alloc(NULL, 16)) return -1;
	rt_fs_conf_init(&hyperflash_conf);

	int flashBuffSize = FLASH_BUFF_SIZE * sizeof(short int);
	void *flashBuffer = rt_alloc(RT_ALLOC_PERIPH, flashBuffSize);
	if(flashBuffer == NULL) {
#ifdef VERBOSE
		printf("Error Allocating flashBuffSize\n");
#endif
		return -1;
	}

	short unsigned int *checksumBuff = rt_alloc(RT_ALLOC_L2_CL_DATA, flashBuffSize);
	if(checksumBuff == NULL) {
#ifdef VERBOSE
		printf("Error Allocating checksumBuff\n"); 
#endif
		return -1;
	}

	hyperflash_conf.flash_conf.type = RT_FLASH_TYPE_HYPER;
	hyperflash_conf.flash_conf.id = 0;

	fs = rt_fs_mount(NULL, &hyperflash_conf, NULL);
	if(fs == NULL) handle_error();

	for(int i=0; i<NWEIGTHS; i++) {

		/* --------------- Load Weights --------------- */
		unsigned int checksumF = 0;
		unsigned int checksumR = 0;

		// Flash memory opening files for all the Layer's weights
		file = rt_fs_open(fs, L3_weights_files[i], 0, NULL);
		if(file == NULL) {
#ifdef VERBOSE
			printf("Error Opening file: %s\n", L3_weights_files[i]); 
#endif
			return -1;
		}

		L3_sizes[i] = file->size;

		// L3 Memory allocation for all the Layer's weights (i.e.HyperRam)
		L3_weights[i] = (short int *) rt_hyperram_alloc(hyperram, L3_sizes[i]);
		if(L3_weights[i] == NULL) {
#ifdef VERBOSE
			printf("Error Allocating L3: Layer %d\n", i+1); 
#endif
			return -1;
		}

		unsigned int rdDone = 0;
		// loop on chunk in file
		while(rdDone < (L3_sizes[i] / sizeof(short int))) { 

			// read from HyperFlash
			int size = rt_fs_read(file, flashBuffer, flashBuffSize, NULL);

			// write to HyperRam
			rt_hyperram_write(hyperram, flashBuffer, L3_weights[i]+rdDone, size, NULL);

			// checksum 
			rt_hyperram_read(hyperram, checksumBuff, L3_weights[i]+rdDone, size, NULL);

			short unsigned int *ptr = (short unsigned int *) flashBuffer;
			for(int j=0; j<(int)(size / sizeof(short unsigned int)); j++) {
				checksumF += (unsigned int) ptr[j];
				checksumR += (unsigned int) checksumBuff[j];
			}

			rdDone += size / sizeof(short int);
		}

		if(checksumF != L3_weights_GT[i] || checksumR != L3_weights_GT[i]) {
#ifdef VERBOSE
			printf("Error Checksum Weight %d: %u %u [%u]\n", i+1, checksumF, checksumR, L3_weights_GT[i]);
#endif
			return -1;
		}

		/* ---------------- Load Biases --------------- */
		// flash Memory opening files for all the Layer's biases
		file = rt_fs_open(fs, L3_bias_files[i], 0, NULL);
		if(file == NULL) {
#ifdef VERBOSE
			printf("Error Opening file: %s\n", L3_bias_files[i]); 
#endif
			return -1;
		}

		L2_bias_sizes[i] = file->size;

		if(L2_bias_sizes[i] > FLASH_BUFF_SIZE*sizeof(short int)) {
#ifdef VERBOSE
			printf("Error Bias size exceeding maximum: %d[B]\n", L2_bias_sizes[i]); 
#endif
			return -1;
		}

		L2_bias[i] = rt_alloc(RT_ALLOC_L2_CL_DATA, L2_bias_sizes[i]);

		// read from HyperFlash (min 4 bytes)
		rt_fs_read(file, L2_bias[i], L2_bias_sizes[i], NULL);

		unsigned int checksumB = 0;
		short unsigned int *ptr = (short unsigned int *) L2_bias[i];
		for(unsigned int j=0; j<(L2_bias_sizes[i] / sizeof(short int)); j++){
			checksumB += (unsigned int) ptr[j];
		}

		if(checksumB != L3_biases_GT[i]) {
#ifdef VERBOSE
			printf("Error Checksum Bias %d: %u [%u]\n", i+1, checksumB, L3_biases_GT[i]);
#endif
			return -1;
		}
	}

	// free L2 temporary buffers
	rt_free(RT_ALLOC_PERIPH, flashBuffer, flashBuffSize);
	rt_free(RT_ALLOC_L2_CL_DATA, checksumBuff, flashBuffSize);

	// close HyperFlash 
	rt_fs_unmount(fs, NULL);

#ifdef VERBOSE
	printf("Load HFlash -> HRam:\t\t\tOk\n");
#endif

/* -------------------------------------------------------------------------- */


/* --------------------------- MEMORY ALLOCATION ---------------------------- */

	for(int i=0; i<NLAYERS; i++)
		outputSizesB[i] = outCh[i] * outW[i] * outH[i] * sizeof(short int);

	Norm_Factor[0] = Q_Factor[0]+NORM_INPUT-NORM_ACT;
	Norm_Factor[1] = Q_Factor[1];
	Norm_Factor[2] = Q_Factor[2];
	Norm_Factor[3] = Q_Factor[3];
	Norm_Factor[4] = Q_Factor[4];
	Norm_Factor[5] = Q_Factor[5];
	Norm_Factor[6] = Q_Factor[6];
	Norm_Factor[7] = Q_Factor[7];
	Norm_Factor[8] = Q_Factor[8];
	Norm_Factor[9] = Q_Factor[9];
	Norm_Factor[10] = Q_Factor[10];
	Norm_Factor[11] = Q_Factor[11];

	for(int i=0; i<NUM_L2_BUFF; i++) {
		L2_base[i] = rt_alloc(RT_ALLOC_L2_CL_DATA, L2_buffers_size[i]);
		L2_next_free[i] = L2_base[i];

#ifdef VERBOSE
		printf("L2 Buffer alloc\t%dB\t@ 0x%08x:\t%s\n", L2_buffers_size[i], (unsigned int)L2_base[i], L2_base[i]?"Ok":"Failed");
#endif
		if(L2_base[i] == NULL) return -1;
	}

	// allocate the memory of L2 for the image buffer
	int image_size_bytes = MAX(CAM_CROP_W*CAM_CROP_H*sizeof(short int), CAM_FULLRES_W*CAM_FULLRES_H*sizeof(unsigned char));
	L2_image = rt_alloc(RT_ALLOC_L2_CL_DATA, image_size_bytes);
#ifdef VERBOSE
	printf("L2 Image alloc\t%dB\t@ 0x%08x:\t%s\n", image_size_bytes, (unsigned int) L2_image, L2_image?"Ok":"Failed");
#endif
	if(L2_image == NULL) return -1;

	// power the cluster up. Must be powered-on before allocating L1
	rt_cluster_mount(MOUNT, CID, 0, NULL);

	// allocate some stacks for cluster in L1, rt_nb_pe returns how many cores exist
	void *stacks = rt_alloc(RT_ALLOC_CL_DATA, STACK_SIZE*rt_nb_pe());
#ifdef VERBOSE 
	printf("L1 Stack alloc\t%dB\t@ 0x%08x:\t%s\n", STACK_SIZE*rt_nb_pe(), (unsigned int) stacks, stacks?"Ok":"Failed");
#endif
	if(stacks == NULL) return -1;
	
	PULP_Dronet_L1_Memory = rt_alloc(RT_ALLOC_CL_DATA, _PULP_Dronet_L1_Memory_SIZE);
#ifdef VERBOSE
	printf("L1 Buffer alloc\t%dB\t@ 0x%08x:\t%s\n", _PULP_Dronet_L1_Memory_SIZE, (unsigned int) PULP_Dronet_L1_Memory, PULP_Dronet_L1_Memory?"Ok":"Failed");
 #endif
	if(PULP_Dronet_L1_Memory == NULL) return -1;

/* -------------------------------------------------------------------------- */


/* --------------------------- SPIM CONFIGURATION --------------------------- */

#ifdef SPI_COMM
	// configure the SPI device
	rt_spim_conf_t spim_conf;
	// get default configuration
	rt_spim_conf_init(&spim_conf);
	spim_conf.max_baudrate = 2000000;
	spim_conf.id = 1; 
	spim_conf.cs = 0;
	spim_conf.wordsize = RT_SPIM_WORDSIZE_8;

	// open the device
	rt_spim_t *spim = rt_spim_open(NULL, &spim_conf, NULL);
#ifdef VERBOSE
	printf("SPI Master opening:\t\t\t%s\n", spim?"Ok":"Failed");
#endif
	if(spim == NULL) return -1;

#endif // SPI_COMM

/* -------------------------------------------------------------------------- */


/* -------------------------- CAMERA CONFIGURATION -------------------------- */

	rt_cam_conf_t cam_conf;

	cam_conf.type				= RT_CAM_TYPE_HIMAX;
	cam_conf.resolution 		= QVGA;
	cam_conf.format 			= HIMAX_MONO_COLOR;
	cam_conf.fps 				= fps30;
	cam_conf.slice_en 			= DISABLE;
	cam_conf.shift 				= 0;
	cam_conf.frameDrop_en 		= DISABLE;
	cam_conf.frameDrop_value 	= 0;
	cam_conf.cpiCfg 			= UDMA_CHANNEL_CFG_SIZE_8;
#if PLATFORM==2 // GAPuino
	cam_conf.control_id			= 1;
#else // PULP-Shield or GV-SoC
	cam_conf.control_id			= 0;
#endif
	cam_conf.id					= 0;

	camera = rt_camera_open(NULL, &cam_conf, 0);

#ifdef VERBOSE
		printf("HiMax camera opening:\t\t\t%s\n", camera?"Ok":"Failed");
#endif
	if(camera == NULL) return -1;

	himaxRegWrite(camera, IMG_ORIENTATION,	0x00);	//	Img orientation		[Def: 0x10]
	himaxRegWrite(camera, AE_TARGET_MEAN, 	0x4E);	//	AE target mean 		[Def: 0x3C]
	himaxRegWrite(camera, AE_MIN_MEAN,		0x1C);	//	AE min target mean 	[Def: 0x0A]
	himaxRegWrite(camera, MAX_AGAIN_FULL,	0x02);	//	Max AGAIN full res 	[Def: 0x00]
	himaxRegWrite(camera, MIN_AGAIN, 		0x00);	//	Min AGAIN 			[Def: 0x00]
	himaxRegWrite(camera, BLC_TGT, 			0x20);	//	Black level target 	[Def: 0x20]
	himaxRegWrite(camera, ANALOG_GAIN,		0x00);	//	Analog Global Gain 	[Def: 0x00]
	himaxRegWrite(camera, DIGITAL_GAIN_H, 	0x03);	//	Digital Gain High 	[Def: 0x01]
	himaxRegWrite(camera, DIGITAL_GAIN_L, 	0xFC);	//	Digital Gain Low 	[Def: 0x00]
	himaxRegWrite(camera, MAX_DGAIN,		0xF0);	//	Max DGAIN 			[Def: 0xC0]
	himaxRegWrite(camera, MIN_DGAIN, 		0x60);	//	Min DGAIN 			[Def: 0x40]
	himaxRegWrite(camera, SINGLE_THR_HOT, 	0xFF);	//	single hot px th 	[Def: 0xFF]
	himaxRegWrite(camera, SINGLE_THR_COLD,	0xFF);	//	single cold px th 	[Def: 0xFF]

	rt_cam_control(camera, CMD_INIT, 0);

#if defined(CROPPING) && CROPPING == 0
	rt_img_slice_t slicer;
	slicer.slice_ll.x = LL_X;
	slicer.slice_ll.y = LL_Y;
	slicer.slice_ur.x = UR_X;
	slicer.slice_ur.y = UR_Y;
	
	rt_cam_control(camera, CMD_START, 0);
	rt_cam_control(camera, CMD_SLICE, &slicer);
#endif

	// wait the camera to setup
	if(rt_platform() == ARCHI_PLATFORM_BOARD)
		rt_time_wait_us(1000000);

/* -------------------------------------------------------------------------- */


/* ------------------------- CAMERA 1st ACQUISITION ------------------------- */

	// grab the first frame in advance, because this requires some extra time
	enqueue_capture();
	
	// wait on input image transfer 
	while(imgTransferDone==0) {
		rt_event_yield(NULL);
	}

/* -------------------------------------------------------------------------- */


/* ----------------------------- RUN PULP-DRONET ---------------------------- */

	volatile int iter = 0;

#ifndef DATASET_TEST
	while(1) {
#endif

#ifdef PROFILE_FC
		rt_perf_t perf_fc;
		rt_perf_init(&perf_fc);
		rt_perf_conf(&perf_fc, (1<<RT_PERF_CYCLES));
		rt_perf_reset(&perf_fc);
		rt_perf_start(&perf_fc);
#endif

		// wait on input image transfer 
		while(imgTransferDone==0) {
			rt_event_yield(NULL);
		}
		imgTransferDone=0;

		event_cluster = rt_event_get_blocking(NULL);
		event_capture = rt_event_get(NULL, enqueue_capture, NULL);

		// execute the function "RunPULPDronet" on the cluster
		rt_cluster_call(NULL, CID, (void *) RunPULPDronet, NULL, stacks, STACK_SIZE, STACK_SIZE, rt_nb_pe(), event_cluster);

		rt_event_wait(event_cluster);

#ifdef SPI_COMM
		// SPI write out result
		rt_spim_send(spim, SPIM_tx, SPIM_BUFFER*8, RT_SPIM_CS_AUTO, NULL);
#endif

#ifdef PROFILE_FC
		rt_perf_stop(&perf_fc);
		rt_perf_save(&perf_fc);
		printf("FC Cycles:\t\t%d\n", rt_perf_get(&perf_fc, RT_PERF_CYCLES));
#endif

#ifdef VERBOSE
	#ifdef DEBUG
		printf("Result[steer][coll]:\t%f\t%f\n", fixed2float(SPIM_tx[0], NORM_ACT), fixed2float(SPIM_tx[1], NORM_ACT));
	#else
		printf("Result[steer][coll]:\t%d\t%d\n", SPIM_tx[0], SPIM_tx[1]);
	#endif
#endif

		iter++;
#ifndef DATASET_TEST
	}
#endif

/* -------------------------------------------------------------------------- */


/* --------------------------- FINAL FREE/CLOSE ----------------------------- */

	// close camera module
	rt_camera_close(camera, 0);

#ifdef SPI_COMM
	// close SPI interface
	rt_spim_close(spim, NULL);
#endif

	rt_free(RT_ALLOC_L2_CL_DATA, L2_image, CAM_CROP_W*CAM_CROP_H*sizeof(short int));

	for(int i=0; i<NUM_L2_BUFF; i++)
		rt_free(RT_ALLOC_L2_CL_DATA, L2_base[i], L2_buffers_size[i]);

	// free HyperRam
	for(int i=0; i<NWEIGTHS; i++) {
		rt_hyperram_free(hyperram, L3_weights[i], L3_sizes[i]);
		rt_free(RT_ALLOC_L2_CL_DATA, L2_bias[i], L2_bias_sizes[i]);
	}

	// close HyperRam
	rt_hyperram_close(hyperram, NULL);

/* -------------------------------------------------------------------------- */

	// power the cluster down
	rt_cluster_mount(UNMOUNT, CID, 0, NULL);

	return 0;
}