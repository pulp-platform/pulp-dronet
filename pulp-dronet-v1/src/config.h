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
 * File:    config.h                                                          *
 * Author:  Daniele Palossi <dpalossi@iis.ee.ethz.ch>                         *
 * Date:    10.04.2019                                                        *
 *----------------------------------------------------------------------------*/


#ifndef PULP_DRONET_CONFIG
#define PULP_DRONET_CONFIG

/****************************** USER PARAMETERS *******************************/
// #define DATASET_TEST				// Enable if you test the Dataset (single iteration)
// #define VERBOSE					// Enables additional information
// #define CHECKSUM					// Enables correctness check per layer
// #define PROFILE_CL				// Profiling execution from the Cluster
// #define PROFILE_FC				// Profiling execution from the Fabric Ctrl
#define PLATFORM		1			// Select 1 for PULP-Shield/GV-SoC or 2 for GAPuino
#define CROPPING 		1			// Image cropping enable: 0 in HW, 1 in SW
#define SPI_COMM					// Enables SPI communication
#define CAM_FULLRES_W	324			// HiMax full width 324
#define CAM_FULLRES_H	244			// HiMax full height 244
#define CAM_CROP_W		200			// Cropped camera width 
#define CAM_CROP_H		200			// Cropped camera height 
#define	NORM_INPUT		8			// Input image Norm Factor [Default Q8.8]
#define NORM_ACT		11			// Activations Norm Factor [Default Q5.11]
/***************************** DEBUGGING SUPPORT ******************************/
#ifdef DEBUG						// Debug mode requires -Os in the Makefile
// #define DUMP_I			0		// Dump input FMs: [0 to 18] - 18 means all
// #define DUMP_O			0		// Dump output FMs: [0 to 18] - 18 means all
// #define DUMP_W			0		// Dump weights: [0 to 18] - 18 means all
// #define DUMP_B			0		// Dump biases: [0 to 18] - 18 means all
// #define DUMP_T			0		// Dump data type: 0 int, 1 float
#endif

/* ---------------------- PULP-DroNet Operative Points ---------------------- *
 * 	Most energy-efficient:	VDD@1.0V	FC 50MHz	CL 100MHz	(6fps@64mW)	  *
 *	Highest performance:	VDD@1.2V	FC 250MHz	CL 250MHz	(18fps@272mW) *
 * -------------------------------------------------------------------------- */

#define VOLTAGE_SOC		1000		// SoC voltage level (1000, 1050, 1100, 1150, 1200 mV)
#define FREQUENCY_FC	50000000	// Fabric Ctrl target frequency [Hz]
#define FREQUENCY_CL	100000000	// Cluster target frequency [Hz]
/******************************************************************************/

// LAYER			ID		InCh	InW		InH		OutCh	OutW	OutH	KerW	KerH	Stride	Bias[B] Weights[B]
// 5x5ConvMax_1		0		1		200		200		32		50		50		5		5		2,2		64		1600
// ReLU_1			1		32		50		50		32		50		50		--		--		--		--		--
// 3x3ConvReLU_2	2		32		50		50		32		25		25		3		3		2		64		18432
// 3x3Conv_3		3		32		25		25		32		25		25		3		3		1		64		18432
// 1x1Conv_4		4		32		50		50		32		25		25		1		1		2		64		2048
// Add_1			5		32		25		25		32		25		25		--		--		--		--		--
// ReLU_2			6		32		25		25		32		25		25		--		--		--		--		--
// 3x3ConvReLU_5	7		32		25		25		64		13		13		3		3		2		128		36864
// 3x3Conv_6		8		64		13		13		64		13		13		3		3		1		128		73728
// 1x1Conv_7		9		32		25		25		64		13		13		1		1		2		128		4096
// Add_2 			10		64		13		13		64		13		13		--		--		--		--		--
// ReLU_3			11		64		13		13		64		13		13		--		--		--		--		--
// 3x3ConvReLU_8	12		64		13		13		128		7		7		3		3		2		256		147456
// 3x3Conv_9		13		128		7		7		128		7		7		3		3		1		256		294912
// 1x1Conv_10		14		64		13		13		128		7		7		1		1		2		256		16384
// AddReLU_3		15		128		7		7		128		7		7		--		--		--		--		--
// Dense_1			16		128		7		7		1		1		1		7		7		1		2		12544
// Dense_2			17		128		7		7		1		1		1		7		7		1		2		12544

/***************************** PRIVATE PARAMETERS *****************************/
#define STACK_SIZE		1200		// Stack size per Core
#define MOUNT			1			// Cluster mount command
#define UNMOUNT			0			// Cluster unmount command
#define CID				0			// Cluster ID
#define FLASH_BUFF_SIZE	128 		// Safe to keep this <= 256 Bytes
#define NLAYERS			18			// Overall number of layers (ReLu, Add, Conv, Dense)
#define NWEIGTHS		12			// Number of Conv Weights
#define SPIM_BUFFER		4			// SPI master buffer size [Bytes]
#define NORM_BIAS_DENSE	NORM_ACT	// Normalization Factor for the Biases of dense layers
#define NUM_L2_BUFF		2			// Number of L2 buffers 
#define	CROPPING_X		1			// Cropping area X (Horizontal/Width): 0=Left, 1=Central, 2=Right
#define	CROPPING_Y		2			// Cropping area Y (Vertical/Height): 0=Top, 1=Central, 2=Bottom
/******************************************************************************/

/****************************** Cropping Setting *******************************
 * PULP-DroNet default cropping is Central/Bottom (X/Y)						   *
 *																			   *
 *											(X)								   *
 *						|	0:Left	|	1:Central	|	2:Right		|		   *
 *						 ___________________________________________		   *
 *			0:Top		|___________|_______________|_______________|		   *
 *	(Y)		1:Central	|___________|_______________|_______________|		   *
 *			2:Bottom	|___________|____Default____|_______________|		   *
 *																			   *
 ******************************************************************************/

#if CROPPING_X==0 		// X Left [0-200]
#define LL_X			0 								// left x coordinate 0
#elif CROPPING_X==1		// X Central [62-262]
#define LL_X			((CAM_FULLRES_W-CAM_CROP_W)/2) 	// left x coordinate 62
#elif CROPPING_X==2		// X Right [124-324]
#define LL_X			(CAM_FULLRES_W-CAM_CROP_W) 		// left x coordinate 124
#endif

#if CROPPING_Y==0 		// Y Top [0-200]
#define LL_Y			0								// up y coordinate 0
#elif CROPPING_Y==1 	// Y Central [22-222]
#define LL_Y			((CAM_FULLRES_H-CAM_CROP_H)/2)	// up y coordinate 22
#elif CROPPING_Y==2 	// Y Bottom [44-244]
#define LL_Y			(CAM_FULLRES_H-CAM_CROP_H)		// up y coordinate 44
#endif

#define UR_X			CAM_CROP_W+LL_X					// right x coordinate
#define UR_Y			CAM_CROP_H+LL_Y 				// bottom y coordinate

/******************************************************************************/

#if !defined(CROPPING) || CROPPING==1
#define CAM_WIDTH		CAM_FULLRES_W
#define CAM_HEIGHT		CAM_FULLRES_H
#else
#define CAM_WIDTH		CAM_CROP_W
#define CAM_HEIGHT		CAM_CROP_H
#endif

// preventing binary size inflating L2
#if defined(CHECKSUM)
#undef PROFILE_FC
#undef PROFILE_CL
#endif
#if defined(PROFILE_FC)
#undef CHECKSUM
#undef PROFILE_CL
#endif
#if defined(PROFILE_CL)
#undef CHECKSUM
#undef PROFILE_FC
#endif

// LUT mapping conv layer among all layers, IN[0-17] OUT[0-11], -1 error
const int			LAYERS_MAPPING_LUT[] = {
	0,							// 0	5x5ConvMax_1			
	-1,							// 1	ReLU_1				
	1,							// 2	3x3ConvReLU_2		
	2,							// 3	3x3Conv_3			
	3,							// 4	1x1Conv_4			
	-1,							// 5	Add_1				
	-1,							// 6	ReLU_2				
	4,							// 7	3x3ConvReLU_5		
	5,							// 8	3x3Conv_6			
	6,							// 9	1x1Conv_7			
	-1,							// 10	Add_2 				
	-1,							// 11	ReLU_3				
	7,							// 12	3x3ConvReLU_8		
	8,							// 13	3x3Conv_9			
	9,							// 14	1x1Conv_10			
	-1,							// 15	AddReLU_3			
	10,							// 16	Dense_1				
	11							// 17	Dense_2	
 };

/******************************************************************************/

/* ----------------------------- L2 Buffer Sizes ---------------------------- */
const int			L2_buffers_size[NUM_L2_BUFF] = {320000, 61632};

/* --------------------------- Input Channel Sizes -------------------------- */
const int			inCh[] = {
	1,							// 0	5x5ConvMax_1
	32,							// 1	ReLU_1
	32,							// 2	3x3ConvReLU_2
	32,							// 3	3x3Conv_3
	32,							// 4	1x1Conv_4
	32,							// 5	Add_1
	32,							// 6	ReLU_2
	32,							// 7	3x3ConvReLU_5
	64,							// 8	3x3Conv_6
	32,							// 9	1x1Conv_7
	64,							// 10	Add_2
	64,							// 11	ReLU_3
	64,							// 12	3x3ConvReLU_8
	128,						// 13	3x3Conv_9
	64,							// 14	1x1Conv_10
	128,						// 15	AddReLU_3
	128,						// 16	Dense_1
	128							// 17	Dense_2
};

/* ------------------------- Input Feature Map Width ------------------------ */
const int			inW[] = {
	200,						// 0	5x5ConvMax_1
	50,							// 1	ReLU_1
	50,							// 2	3x3ConvReLU_2
	25,							// 3	3x3Conv_3
	50,							// 4	1x1Conv_4
	25,							// 5	Add_1
	25,							// 6	ReLU_2
	25,							// 7	3x3ConvReLU_5
	13,							// 8	3x3Conv_6
	25,							// 9	1x1Conv_7
	13,							// 10	Add_2
	13,							// 11	ReLU_3
	13,							// 12	3x3ConvReLU_8
	7,							// 13	3x3Conv_9
	13,							// 14	1x1Conv_10
	7,							// 15	AddReLU_3
	7,							// 16	Dense_1
	7							// 17	Dense_2
};

/* ------------------------ Input Feature Map Height ------------------------ */
const int			inH[] = {
	200,						// 0	5x5ConvMax_1
	50,							// 1	ReLU_1
	50,							// 2	3x3ConvReLU_2
	25,							// 3	3x3Conv_3
	50,							// 4	1x1Conv_4
	25,							// 5	Add_1
	25,							// 6	ReLU_2
	25,							// 7	3x3ConvReLU_5
	13,							// 8	3x3Conv_6
	25,							// 9	1x1Conv_7
	13,							// 10	Add_2
	13,							// 11	ReLU_3
	13,							// 12	3x3ConvReLU_8
	7,							// 13	3x3Conv_9
	13,							// 14	1x1Conv_10
	7,							// 15	AddReLU_3
	7,							// 16	Dense_1
	7							// 17	Dense_2
};

/* ------------------------ Conv Layers Filter Width ------------------------ */
const int			kerW[] = {
	5,							// 0	5x5ConvMax_1
	3,							// 2	3x3ConvReLU_2
	3,							// 3	3x3Conv_3
	1,							// 4	1x1Conv_4
	3,							// 7	3x3ConvReLU_5
	3,							// 8	3x3Conv_6
	1,							// 9	1x1Conv_7
	3,							// 12	3x3ConvReLU_8
	3,							// 13	3x3Conv_9
	1,							// 14	1x1Conv_10
	7,							// 16	Dense_1
	7							// 17	Dense_2
};

/* ----------------------- Conv Layers Filter Height ----------------------- */
const int			kerH[] = {
	5,							// 0	5x5ConvMax_1
	3,							// 2	3x3ConvReLU_2
	3,							// 3	3x3Conv_3
	1,							// 4	1x1Conv_4
	3,							// 7	3x3ConvReLU_5
	3,							// 8	3x3Conv_6
	1,							// 9	1x1Conv_7
	3,							// 12	3x3ConvReLU_8
	3,							// 13	3x3Conv_9
	1,							// 14	1x1Conv_10
	7,							// 16	Dense_1
	7							// 17	Dense_2
};


/* -------------------------- Output Channel Sizes -------------------------- */
const int			outCh[] = {
	32,							// 0	5x5ConvMax_1
	32,							// 1	ReLU_1
	32,							// 2	3x3ConvReLU_2
	32,							// 3	3x3Conv_3
	32,							// 4	1x1Conv_4
	32,							// 5	Add_1
	32,							// 6	ReLU_2
	64,							// 7	3x3ConvReLU_5
	64,							// 8	3x3Conv_6
	64,							// 9	1x1Conv_7
	64,							// 10	Add_2
	64,							// 11	ReLU_3
	128,						// 12	3x3ConvReLU_8
	128,						// 13	3x3Conv_9
	128,						// 14	1x1Conv_10
	128,						// 15	AddReLU_3
	1,							// 16	Dense_1
	1							// 17	Dense_2
};

/* ------------------------ Output Feature Map Width ------------------------ */
const int			outW[] = {
	50,							// 0	5x5ConvMax_1
	50,							// 1	ReLU_1
	25,							// 2	3x3ConvReLU_2
	25,							// 3	3x3Conv_3
	25,							// 4	1x1Conv_4
	25,							// 5	Add_1
	25,							// 6	ReLU_2
	13,							// 7	3x3ConvReLU_5
	13,							// 8	3x3Conv_6
	13,							// 9	1x1Conv_7
	13,							// 10	Add_2
	13,							// 11	ReLU_3
	7,							// 12	3x3ConvReLU_8
	7,							// 13	3x3Conv_9
	7,							// 14	1x1Conv_10
	7,							// 15	AddReLU_3
	1,							// 16	Dense_1
	1							// 17	Dense_2
};

/* ------------------------ Output Feature Map Width ------------------------ */
const int			outH[] = {
	50,							// 0	5x5ConvMax_1
	50,							// 1	ReLU_1
	25,							// 2	3x3ConvReLU_2
	25,							// 3	3x3Conv_3
	25,							// 4	1x1Conv_4
	25,							// 5	Add_1
	25,							// 6	ReLU_2
	13,							// 7	3x3ConvReLU_5
	13,							// 8	3x3Conv_6
	13,							// 9	1x1Conv_7
	13,							// 10	Add_2
	13,							// 11	ReLU_3
	7,							// 12	3x3ConvReLU_8
	7,							// 13	3x3Conv_9
	7,							// 14	1x1Conv_10
	7,							// 15	AddReLU_3
	1,							// 16	Dense_1
	1							// 17	Dense_2
};

/* ------------------------- L3 Weights File Names -------------------------- */
const char *		L3_weights_files[] = {
	"weights_conv2d_1.hex",		// 0	5x5ConvMax_1	1600	Bytes
	"weights_conv2d_2.hex",		// 2	3x3ConvReLU_2	18432	Bytes
	"weights_conv2d_3.hex",		// 3	3x3Conv_3		18432	Bytes
	"weights_conv2d_4.hex",		// 4	1x1Conv_4		2048	Bytes
	"weights_conv2d_5.hex",		// 7	3x3ConvReLU_5	36864	Bytes
	"weights_conv2d_6.hex",		// 8	3x3Conv_6		73728	Bytes
	"weights_conv2d_7.hex",		// 9	1x1Conv_7		4096	Bytes
	"weights_conv2d_8.hex",		// 12	3x3ConvReLU_8	147456	Bytes
	"weights_conv2d_9.hex",		// 13	3x3Conv_9		294912	Bytes
	"weights_conv2d_10.hex",	// 14	1x1Conv_10		16384	Bytes
	"weights_dense_1.hex",		// 16	Dense_1			12544	Bytes
	"weights_dense_2.hex"		// 17	Dense_2			12544	Bytes
};

/* -------------------------- L3 Biases File Names -------------------------- */
const char *		L3_bias_files[] = {
	"bias_conv2d_1.hex",		// 0	5x5ConvMax_1	64		Bytes
	"bias_conv2d_2.hex",		// 2	3x3ConvReLU_2	64		Bytes
	"bias_conv2d_3.hex",		// 3	3x3Conv_3		64		Bytes
	"bias_conv2d_4.hex",		// 4	1x1Conv_4		64		Bytes
	"bias_conv2d_5.hex",		// 7	3x3ConvReLU_5	128		Bytes
	"bias_conv2d_6.hex",		// 8	3x3Conv_6		128		Bytes
	"bias_conv2d_7.hex",		// 9	1x1Conv_7		128		Bytes
	"bias_conv2d_8.hex",		// 12	3x3ConvReLU_8	256		Bytes
	"bias_conv2d_9.hex",		// 13	3x3Conv_9		256		Bytes
	"bias_conv2d_10.hex",		// 14	1x1Conv_10		256		Bytes
	"bias_dense_1.hex",			// 16	Dense_1			2		Bytes
	"bias_dense_2.hex"			// 17	Dense_2			2		Bytes
};

/* ----------------------- Weights Ground Truth (GT) ------------------------ */
const unsigned int	L3_weights_GT[NWEIGTHS] = {
	25985189,					// 0	5x5ConvMax_1
	171247369,					// 2	3x3ConvReLU_2
	215325794,					// 3	3x3Conv_3
	31883544,					// 4	1x1Conv_4
	198666713,					// 7	3x3ConvReLU_5
	401025883,					// 8	3x3Conv_6
	63738646,					// 9	1x1Conv_7
	264630086,					// 12	3x3ConvReLU_8
	194687313,					// 13	3x3Conv_9
	281798119,					// 14	1x1Conv_10
	204663510,					// 16	Dense_1
	244332381					// 17	Dense_2
};

/* ------------------------ Biases Ground Truth (GT) ------------------------ */
const unsigned int	L3_biases_GT[NWEIGTHS] = {
	1484287,					// 0	5x5ConvMax_1
	1364354,					// 2	3x3ConvReLU_2
	1369951,					// 3	3x3Conv_3
	1045420,					// 4	1x1Conv_4
	3869389,					// 7	3x3ConvReLU_5
	3442234,					// 8	3x3Conv_6
	3245463,					// 9	1x1Conv_7
	7635667,					// 12	3x3ConvReLU_8
	8031559,					// 13	3x3Conv_9
	7469135,					// 14	1x1Conv_10
	0,							// 16	Dense_1
	944							// 17	Dense_2
};

/* --------------------- Quantization Factor per layer ---------------------- */
const int			Q_Factor[NWEIGTHS] = {
	12,							// 0	5x5ConvMax_1
	14,							// 2	3x3ConvReLU_2
	14,							// 3	3x3Conv_3
	7,							// 4	1x1Conv_4
	14,							// 7	3x3ConvReLU_5
	14,							// 8	3x3Conv_6
	14,							// 9	1x1Conv_7
	14,							// 12	3x3ConvReLU_8
	14,							// 13	3x3Conv_9
	12,							// 14	1x1Conv_10
	11,							// 16	Dense_1
	11 							// 17	Dense_2
};

#ifdef CHECKSUM
/* --------------------- Layer output Ground Truth (GT) --------------------- *
 * for:	pulp-dronet/dataset/Himax_Dataset/test_2/frame_22.pgm						  *
 * -------------------------------------------------------------------------- */
const unsigned int	Layer_GT[NLAYERS] = {
	3583346007,					// 0	5x5ConvMax_1
	33640519,					// 1	ReLU_1
	3054757,					// 2	3x3ConvReLU_2
	969723858,					// 3	3x3Conv_3
	733574305,					// 4	1x1Conv_4
	961234035,					// 5	Add_1
	5702624,					// 6	ReLU_2
	800684,						// 7	3x3ConvReLU_5
	568751660,					// 8	3x3Conv_6
	536587146,					// 9	1x1Conv_7
	562176438,					// 10	Add_2
	2426322,					// 11	ReLU_3
	310799,						// 12	3x3ConvReLU_8
	400248723,					// 13	3x3Conv_9
	382744684,					// 14	1x1Conv_10
	34510,						// 15	AddReLU_3
	193,						// 16	Dense_1
	59518						// 17	Dense_2
};
#endif // CHECKSUM


#endif // PULP_DRONET_CONFIG