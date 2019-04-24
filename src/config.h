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
// #define VERBOSE						// Enables additional information
// #define CHECKSUM					// Enables correctness check per layer
// #define PROFILE_CL					// Profiling execution from the Cluster
// #define PROFILE_FC					// Profiling execution from the Fabric Ctrl
#define PLATFORM		1			// Select 1 for PULP-Shield/GV-SoC or 2 for GAPuino
#define CROPPING 		1			// Image cropping enable: 0 in HW, 1 in SW
#define SPI_COMM					// Enables SPI communication
#define CAM_FULLRES_W	324			// HiMax full width 324
#define CAM_FULLRES_H	244			// HiMax full height 244
#define CAM_CROP_W		200			// Cropped camera width 
#define CAM_CROP_H		200			// Cropped camera height 
#define	NORM_INPUT		8			// Input image Norm Factor [Default Q8.8]
#define NORM_ACT		11			// Activations Norm Factor [Default Q5.11]

/* ---------------------- PULP-DroNet Operative Points ---------------------- *
 * 	Most energy-efficient:	VDD@1.0V	FC 50MHz	CL 100MHz	(6fps@64mW)	  *
 *	Highest performance:	VDD@1.2V	FC 250MHz	CL 250MHz	(18fps@272mW) *
 * -------------------------------------------------------------------------- */

#define VOLTAGE_SOC		1000		// SoC voltage level (1000, 1050, 1100, 1150, 1200 mV)
#define FREQUENCY_FC	50000000	// Fabric Ctrl target frequency [Hz]
#define FREQUENCY_CL	100000000	// Cluster target frequency [Hz]
/******************************************************************************/

// LAYER			ID		InCh	InSize	OutCh	OutSize	KerSize	Stride	Bias[B] Weights[B]
// 5x5ConvMax_1		1		1		200		32		50		5		2,2		64		1600
// ReLU_1			2		32		50		32		50		--		--		--		--
// 3x3ConvReLU_2	3		32		50		32		25		3		2		64		18432
// 3x3Conv_3		4		32		25		32		25		3		1		64		18432
// 1x1Conv_4		5		32		50		32		25		1		2		64		2048
// Add_1			6		32		25		32		25		--		--		--		--
// ReLU_2			7		32		25		32		25		--		--		--		--
// 3x3ConvReLU_5	8		32		25		64		13		3		2		128		36864
// 3x3Conv_6		9		64		13		64		13		3		1		128		73728
// 1x1Conv_7		10		32		25		64		13		1		2		128		4096
// Add_2 			11		64		13		64		13		--		--		--		--
// ReLU_3			12		64		13		64		13		--		--		--		--
// 3x3ConvReLU_8	13		64		13		128		7		3		2		256		147456
// 3x3Conv_9		14		128		7		128		7		3		1		256		294912
// 1x1Conv_10		15		64		13		128		7		1		2		256		16384
// AddReLU_3		16		128		7		128		7		--		--		--		--
// Dense_1			17		128		7		1		1		7		1		2		12544
// Dense_2			18		128		7		1		1		7		1		2		12544

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

/******************************************************************************/

/* ----------------------------- L2 Buffer Sizes ---------------------------- */
const int			L2_buffers_size[NUM_L2_BUFF] = {320000, 61632};

/* --------------------------- Input Channel Sizes -------------------------- */
const int			inCh[] = {
	1,							// 1	5x5ConvMax_1
	32,							// 2	ReLU_1
	32,							// 3	3x3ConvReLU_2
	32,							// 4	3x3Conv_3
	32,							// 5	1x1Conv_4
	32,							// 6	Add_1
	32,							// 7	ReLU_2
	32,							// 8	3x3ConvReLU_5
	64,							// 9	3x3Conv_6
	32,							// 10	1x1Conv_7
	64,							// 11	Add_2
	64,							// 12	ReLU_3
	64,							// 13	3x3ConvReLU_8
	128,						// 14	3x3Conv_9
	64,							// 15	1x1Conv_10
	128,						// 16	AddReLU_3
	128,						// 17	Dense_1
	128							// 18	Dense_2
};

/* ------------------------- Input Feature Map Sizes ------------------------ */
const int			inSize[] = {
	200,						// 1	5x5ConvMax_1
	50,							// 2	ReLU_1
	50,							// 3	3x3ConvReLU_2
	25,							// 4	3x3Conv_3
	50,							// 5	1x1Conv_4
	25,							// 6	Add_1
	25,							// 7	ReLU_2
	25,							// 8	3x3ConvReLU_5
	13,							// 9	3x3Conv_6
	25,							// 10	1x1Conv_7
	13,							// 11	Add_2
	13,							// 12	ReLU_3
	13,							// 13	3x3ConvReLU_8
	7,							// 14	3x3Conv_9
	13,							// 15	1x1Conv_10
	7,							// 16	AddReLU_3
	7,							// 17	Dense_1
	7							// 18	Dense_2
};

/* -------------------------- Output Channel Sizes -------------------------- */
const int			outCh[] = {
	32,							// 1	5x5ConvMax_1
	32,							// 2	ReLU_1
	32,							// 3	3x3ConvReLU_2
	32,							// 4	3x3Conv_3
	32,							// 5	1x1Conv_4
	32,							// 6	Add_1
	32,							// 7	ReLU_2
	64,							// 8	3x3ConvReLU_5
	64,							// 9	3x3Conv_6
	64,							// 10	1x1Conv_7
	64,							// 11	Add_2
	64,							// 12	ReLU_3
	128,						// 13	3x3ConvReLU_8
	128,						// 14	3x3Conv_9
	128,						// 15	1x1Conv_10
	128,						// 16	AddReLU_3
	1,							// 17	Dense_1
	1							// 18	Dense_2
};

/* ------------------------ Output Feature Map Sizes ------------------------ */
const int			outSize[] = {
	50,							// 1	5x5ConvMax_1
	50,							// 2	ReLU_1
	25,							// 3	3x3ConvReLU_2
	25,							// 4	3x3Conv_3
	25,							// 5	1x1Conv_4
	25,							// 6	Add_1
	25,							// 7	ReLU_2
	13,							// 8	3x3ConvReLU_5
	13,							// 9	3x3Conv_6
	13,							// 10	1x1Conv_7
	13,							// 11	Add_2
	13,							// 12	ReLU_3
	7,							// 13	3x3ConvReLU_8
	7,							// 14	3x3Conv_9
	7,							// 15	1x1Conv_10
	7,							// 16	AddReLU_3
	1,							// 17	Dense_1
	1							// 18	Dense_2
};

/* ------------------------- L3 Weights File Names -------------------------- */
const char *		L3_weights_files[] = {
	"weights_conv2d_1.hex",		// 1	5x5ConvMax_1	1600	Bytes
	"weights_conv2d_2.hex",		// 3	3x3ConvReLU_2	18432	Bytes
	"weights_conv2d_3.hex",		// 4	3x3Conv_3		18432	Bytes
	"weights_conv2d_4.hex",		// 5	1x1Conv_4		2048	Bytes
	"weights_conv2d_5.hex",		// 8	3x3ConvReLU_5	36864	Bytes
	"weights_conv2d_6.hex",		// 9	3x3Conv_6		73728	Bytes
	"weights_conv2d_7.hex",		// 10	1x1Conv_7		4096	Bytes
	"weights_conv2d_8.hex",		// 13	3x3ConvReLU_8	147456	Bytes
	"weights_conv2d_9.hex",		// 14	3x3Conv_9		294912	Bytes
	"weights_conv2d_10.hex",	// 15	1x1Conv_10		16384	Bytes
	"weights_dense_1.hex",		// 17	Dense_1			12544	Bytes
	"weights_dense_2.hex"		// 18	Dense_2			12544	Bytes
};

/* -------------------------- L3 Biases File Names -------------------------- */
const char *		L3_bias_files[] = {
	"bias_conv2d_1.hex",		// 1	5x5ConvMax_1	64		Bytes
	"bias_conv2d_2.hex",		// 3	3x3ConvReLU_2	64		Bytes
	"bias_conv2d_3.hex",		// 4	3x3Conv_3		64		Bytes
	"bias_conv2d_4.hex",		// 5	1x1Conv_4		64		Bytes
	"bias_conv2d_5.hex",		// 8	3x3ConvReLU_5	128		Bytes
	"bias_conv2d_6.hex",		// 9	3x3Conv_6		128		Bytes
	"bias_conv2d_7.hex",		// 10	1x1Conv_7		128		Bytes
	"bias_conv2d_8.hex",		// 13	3x3ConvReLU_8	256		Bytes
	"bias_conv2d_9.hex",		// 14	3x3Conv_9		256		Bytes
	"bias_conv2d_10.hex",		// 15	1x1Conv_10		256		Bytes
	"bias_dense_1.hex",			// 17	Dense_1			2		Bytes
	"bias_dense_2.hex"			// 18	Dense_2			2		Bytes
};

/* ----------------------- Weights Ground Truth (GT) ------------------------ */
const unsigned int	L3_weights_GT[NWEIGTHS] = {
	25985189,					// 1	5x5ConvMax_1
	171247369,					// 3	3x3ConvReLU_2
	215325794,					// 4	3x3Conv_3
	31883544,					// 5	1x1Conv_4
	198666713,					// 8	3x3ConvReLU_5
	401025883,					// 9	3x3Conv_6
	63738646,					// 10	1x1Conv_7
	264630086,					// 13	3x3ConvReLU_8
	194687313,					// 14	3x3Conv_9
	281798119,					// 15	1x1Conv_10
	204663510,					// 17	Dense_1
	244332381					// 18	Dense_2
};

/* ------------------------ Biases Ground Truth (GT) ------------------------ */
const unsigned int	L3_biases_GT[NWEIGTHS] = {
	1484287,					// 1	5x5ConvMax_1
	1364354,					// 3	3x3ConvReLU_2
	1369951,					// 4	3x3Conv_3
	1045420,					// 5	1x1Conv_4
	3869389,					// 8	3x3ConvReLU_5
	3442234,					// 9	3x3Conv_6
	3245463,					// 10	1x1Conv_7
	7635667,					// 13	3x3ConvReLU_8
	8031559,					// 14	3x3Conv_9
	7469135,					// 15	1x1Conv_10
	0,							// 17	Dense_1
	944							// 18	Dense_2
};

/* --------------------- Quantization Factor per layer ---------------------- */
const int			Q_Factor[NWEIGTHS] = {
	12,							// 1	5x5ConvMax_1
	14,							// 3	3x3ConvReLU_2
	14,							// 4	3x3Conv_3
	7,							// 5	1x1Conv_4
	14,							// 8	3x3ConvReLU_5
	14,							// 9	3x3Conv_6
	14,							// 10	1x1Conv_7
	14,							// 13	3x3ConvReLU_8
	14,							// 14	3x3Conv_9
	12,							// 15	1x1Conv_10
	11,							// 17	Dense_1
	11 							// 18	Dense_2
};

#ifdef CHECKSUM
/* --------------------- Layer output Ground Truth (GT) --------------------- *
 * for:	pulp-dronet/dataset/Himax_Dataset/test_2/frame_22.pgm						  *
 * -------------------------------------------------------------------------- */
const unsigned int	Layer_GT[NLAYERS] = {
	3583346007,					// 1	5x5ConvMax_1
	33640519,					// 2	ReLU_1
	3054757,					// 3	3x3ConvReLU_2
	969723858,					// 4	3x3Conv_3
	733574305,					// 5	1x1Conv_4
	961234035,					// 6	Add_1
	5702624,					// 7	ReLU_2
	800684,						// 8	3x3ConvReLU_5
	568751660,					// 9	3x3Conv_6
	536587146,					// 10	1x1Conv_7
	562176438,					// 11	Add_2
	2426322,					// 12	ReLU_3
	310799,						// 13	3x3ConvReLU_8
	400248723,					// 14	3x3Conv_9
	382744684,					// 15	1x1Conv_10
	34510,						// 16	AddReLU_3
	193,						// 17	Dense_1
	59518						// 18	Dense_2
};
#endif // CHECKSUM


#endif // PULP_DRONET_CONFIG