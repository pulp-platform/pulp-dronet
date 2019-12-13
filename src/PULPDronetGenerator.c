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
 * File:    PULPDronetGenerator.c                                             *
 * Author:  Daniele Palossi <dpalossi@iis.ee.ethz.ch>                         *
 * Date:    10.04.2019                                                        *
 *----------------------------------------------------------------------------*/


#include <stdint.h>
#include <stdio.h>
#include "AutoTilerLib.h"
#include "CNN_Generator.h"
#include "config.h"


void PULPDronetConfiguration(unsigned int L1Size) {

	// Always inline user kernels
	SetInlineMode(ALWAYS_INLINE);
	// C symbols used by the AutoTiler
	SetSymbolNames("PULP_Dronet_L1_Memory", "PULP_Dronet_L2_Memory", "PULP_Dronet_KernelDescr", "PULP_Dronet_KernelArgs");
	// L1 and L2 symbols are dynamic
	SetSymbolDynamics();
	// Standard data types are used, we import CNN basic kernels
	SetUsedFilesNames("KernelLibStdTypes.h", 1, "CNN_BasicKernels.h");
	// AutoTiler generated files
	SetGeneratedFilesNames("PULPDronetKernelsInit.c", "PULPDronetKernelsInit.h", "PULPDronetKernels.c", "PULPDronetKernels.h");
	// L1 shared memory given to AutoTiler
	SetL1MemorySize(L1Size);
}


void PULPDronetGenerator() {

/* --------------------------------- LAYER 1 -------------------------------- */
	/* 5x5 Convolution Stride 2, followed by 3x3 Max pooling. Pure SW.
	 * 1 input plane [200x200], 32 output planes [50x50] */
	LargeParOutFeatConvolutionPoolReLU_Hor_fp(
		"LargeParConv_5x5_S2_Max2x2_S2_H_1",	// Name: 		Name
		inCh[0],								// InFeat:		Number Input Channels
		outCh[0],								// OutFeat:		Number Output Channels
		inW[0],									// Width:		Feature Map Width
		inH[0],									// Height:		Feature Map Height
		5,										// FSc:			Filter Selection (1x1, 3x3, 5x5)
		2,										// ConvStride:	Convolution Stride (\1, \2)
		1,										// ConvDoPad:	Padded Convolution (yes=1, no=0)
		0,										// ConvDoReLU:	ReLu after Convolution (yes=1, no=0)
		2,										// FSp: 		Pooling Selection (only 2x2 supported)
		2,										// PoolStride:	Pooling Stride (\1, \2)
		0,										// PoolDoPad:	Padded MaxPooling (not yet used)
		0,										// PoolDoReLU:	ReLu after MaxPooling (yes=1, no=0)
		1);										// DoPool:		MaxPooling Enable (yes=1, no=0)


/* --------------------------------- LAYER 2 -------------------------------- */
	/* ReLU. Pure SW.
	 * 32 input planes [50x50], 32 output planes [50x50] */
	CNN_ReLu_SW_fp(
		"ReLU_SW_1",							// Name: 		Name
		inCh[1],								// InFeat:		Number Input Channels
		outCh[1],								// OutFeat:		Number Output Channels
		inW[1],							 		// Width:		Feature Map Width
		inH[1]);								// Height:		Feature Map Height

	/* 3x3 Convolution Stride 2, followed by ReLU. Pure SW.
	* 32 input planes [50x50], 32 output planes [25x25] */
	MediumParOutFeatConvolutionPoolReLU_fp(
		"MedParConv_3x3_S2_ReLU_2",				// Name: 		Name
		inCh[2],								// InFeat:		Number Input Channels
		outCh[2],								// OutFeat:		Number Output Channels
		inW[2],									// Width:		Feature Map Width
		inH[2],									// Height:		Feature Map Height
		3,										// FSc:			Filter Selection (1x1, 3x3, 5x5)
		2,										// ConvStride:	Convolution Stride (\1, \2) 
		1,										// ConvDoPad:	Padded Convolution (yes=1, no=0)
		1,										// ConvDoReLU:	ReLu after Convolution (yes=1, no=0)
		0,										// FSp: 		Pooling Selection (only 2x2 supported)
		0,										// PoolStride:	Pooling Stride (\1, \2)
		0,										// PoolDoPad:	Padded MaxPooling (not yet used)
		0,										// PoolDoReLU:	ReLu after MaxPooling (yes=1, no=0)
		0);										// DoPool:		MaxPooling Enable (yes=1, no=0)


/* --------------------------------- LAYER 3 -------------------------------- */
	/* 3x3 Convolution. Pure SW.
	 * 32 input planes [25x25], 32 output planes [25x25] */
	MediumParOutFeatConvolutionPoolReLU_fp(
		"MedParConv_3x3_S1_3",					// Name: 		Name
		inCh[3],								// InFeat:		Number Input Channels
		outCh[3],								// OutFeat:		Number Output Channels
		inW[3],									// Width:		Feature Map Width
		inH[3],									// Height:		Feature Map Height
		3,										// FSc:			Filter Selection (1x1, 3x3, 5x5)
		1,										// ConvStride:	Convolution Stride (\1, \2) 
		1,										// ConvDoPad:	Padded Convolution (yes=1, no=0)
		0,										// ConvDoReLU:	ReLu after Convolution (yes=1, no=0)
		0,										// FSp: 		Pooling Selection (only 2x2 supported)
		0,										// PoolStride:	Pooling Stride (\1, \2)
		0,										// PoolDoPad:	Padded MaxPooling (not yet used)
		0,										// PoolDoReLU:	ReLu after MaxPooling (yes=1, no=0)
		0);										// DoPool:		MaxPooling Enable (yes=1, no=0)


/* --------------------------------- LAYER 4 -------------------------------- */
	/* 1x1 Convolution Stride 2. Pure SW.
	 * 32 input planes [50x50], 32 output planes [25x25] */
	MediumParOutFeatConvolutionPoolReLU_fp(
		"MedParConv_1x1_S2_4",					// Name: 		Name
		inCh[4],								// InFeat:		Number Input Channels
		outCh[4],								// OutFeat:		Number Output Channels
		inW[4],									// Width:		Feature Map Width
		inH[4],									// Height:		Feature Map Height
		1,										// FSc:			Filter Selection (1x1, 3x3, 5x5)
		2,										// ConvStride:	Convolution Stride (\1, \2) 
		0,										// ConvDoPad:	Padded Convolution (yes=1, no=0)
		0,										// ConvDoReLU:	ReLu after Convolution (yes=1, no=0)
		0,										// FSp: 		Pooling Selection (only 2x2 supported)
		0,										// PoolStride:	Pooling Stride (\1, \2)
		0,										// PoolDoPad:	Padded MaxPooling (not yet used)
		0,										// PoolDoReLU:	ReLu after MaxPooling (yes=1, no=0)
		0);										// DoPool:		MaxPooling Enable (yes=1, no=0)


/* -------------------------------- ADD RES 1 ------------------------------- */
	/* Matrix Add. Pure SW.
	 * 32 input planes [25x25], 32 output planes [25x25] */
	CNN_MatrixAdd_SW_fp(
		"AddFeatureMaps_SW_1",					// Name: 		Name
		inCh[5],								// InFeat:		Number Input Channels
		outCh[5],								// OutFeat:		Number Output Channels
		inW[5],									// Width:		Feature Map Width
		inH[5]);								// Height:		Feature Map Height


/* --------------------------------- LAYER 5 -------------------------------- */
	/* ReLU. Pure SW.
	 * 32 input planes [25x25], 32 output planes [25x25] */
	CNN_ReLu_SW_fp(
		"ReLU_SW_2",							// Name: 		Name
		inCh[6],								// InFeat:		Number Input Channels
		outCh[6],								// OutFeat:		Number Output Channels
		inW[6],									// Width:		Feature Map Width
		inH[6]);								// Height:		Feature Map Height

	/* 3x3 Convolution Stride 2, followed by ReLU. Pure SW.
	 * 32 input planes [25x25], 64 output planes [13x13] */
	MediumParOutFeatConvolutionPoolReLU_fp(
		"MedParConv_3x3_S2_ReLU_5",				// Name: 		Name
		inCh[7],								// InFeat:		Number Input Channels
		outCh[7],								// OutFeat:		Number Output Channels
		inW[7],									// Width:		Feature Map Width
		inH[7],									// Height:		Feature Map Height
		3,										// FSc:			Filter Selection (1x1, 3x3, 5x5)
		2,										// ConvStride:	Convolution Stride (\1, \2) 
		1,										// ConvDoPad:	Padded Convolution (yes=1, no=0)
		1,										// ConvDoReLU:	ReLu after Convolution (yes=1, no=0)
		0,										// FSp: 		Pooling Selection (only 2x2 supported)
		0,										// PoolStride:	Pooling Stride (\1, \2)
		0,										// PoolDoPad:	Padded MaxPooling (not yet used)
		0,										// PoolDoReLU:	ReLu after MaxPooling (yes=1, no=0)
		0);										// DoPool:		MaxPooling Enable (yes=1, no=0)


/* --------------------------------- LAYER 6 -------------------------------- */
	/* 3x3 Convolution. Pure SW.
	 * 32 input planes [13x13], 64 output planes [13x13] */
	MediumParOutFeatConvolutionPoolReLU_fp(
		"MedParConv_3x3_S1_6",					// Name: 		Name
		inCh[8],								// InFeat:		Number Input Channels
		outCh[8],								// OutFeat:		Number Output Channels
		inW[8],									// Width:		Feature Map Width
		inH[8],									// Height:		Feature Map Height
		3,										// FSc:			Filter Selection (1x1, 3x3, 5x5)
		1,										// ConvStride:	Convolution Stride (\1, \2) 
		1,										// ConvDoPad:	Padded Convolution (yes=1, no=0)
		0,										// ConvDoReLU:	ReLu after Convolution (yes=1, no=0)
		0,										// FSp: 		Pooling Selection (only 2x2 supported)
		0,										// PoolStride:	Pooling Stride (\1, \2)
		0,										// PoolDoPad:	Padded MaxPooling (not yet used)
		0,										// PoolDoReLU:	ReLu after MaxPooling (yes=1, no=0)
		0);										// DoPool:		MaxPooling Enable (yes=1, no=0)


/* --------------------------------- LAYER 7 -------------------------------- */
	/* 1x1 Convolution Stride 2. Pure SW.
	 * 32 input planes [25x25], 64 output planes [13x13] */
	MediumParOutFeatConvolutionPoolReLU_fp(
		"MedParConv_1x1_S2_7",					// Name: 		Name
		inCh[9],								// InFeat:		Number Input Channels
		outCh[9],								// OutFeat:		Number Output Channels
		inW[9],									// Width:		Feature Map Width
		inH[9],									// Height:		Feature Map Height
		1,										// FSc:			Filter Selection (1x1, 3x3, 5x5)
		2,										// ConvStride:	Convolution Stride (\1, \2) 
		0,										// ConvDoPad:	Padded Convolution (yes=1, no=0)
		0,										// ConvDoReLU:	ReLu after Convolution (yes=1, no=0)
		0,										// FSp: 		Pooling Selection (only 2x2 supported)
		0,										// PoolStride:	Pooling Stride (\1, \2)
		0,										// PoolDoPad:	Padded MaxPooling (not yet used)
		0,										// PoolDoReLU:	ReLu after MaxPooling (yes=1, no=0)
		0);										// DoPool:		MaxPooling Enable (yes=1, no=0)


/* -------------------------------- ADD RES 2 ------------------------------- */
	/* Matrix Add. Pure SW.
	 * 64 input planes [13x13], 64 output planes [13x13] */
	CNN_MatrixAdd_SW_fp(
		"AddFeatureMaps_SW_2",					// Name: 		Name
		inCh[10],								// InFeat:		Number Input Channels
		outCh[10],								// OutFeat:		Number Output Channels
		inW[10],								// Width:		Feature Map Width
		inH[10]);								// Height:		Feature Map Height


/* --------------------------------- LAYER 8 -------------------------------- */
	/* ReLU. Pure SW.
	 * 64 input planes [13x13], 64 output planes [13x13] */
	CNN_ReLu_SW_fp(
		"ReLU_SW_3",							// Name: 		Name
		inCh[11],								// InFeat:		Number Input Channels
		outCh[11],								// OutFeat:		Number Output Channels
		inW[11],								// Width:		Feature Map Width
		inH[11]);								// Height:		Feature Map Height

	/* 3x3 Convolution Stride 2, followed by ReLU. Pure SW.
	 * 64 input planes [13x13], 128 output planes [7x7] */
	MediumParOutFeatConvolutionPoolReLU_fp(
		"MedParConv_3x3_S2_ReLU_8",				// Name: 		Name
		inCh[12],								// InFeat:		Number Input Channels
		outCh[12],								// OutFeat:		Number Output Channels
		inW[12],								// Width:		Feature Map Width
		inH[12],								// Height:		Feature Map Height
		3,										// FSc:			Filter Selection (1x1, 3x3, 5x5)
		2,										// ConvStride:	Convolution Stride (\1, \2) 
		1,										// ConvDoPad:	Padded Convolution (yes=1, no=0)
		1,										// ConvDoReLU:	ReLu after Convolution (yes=1, no=0)
		0,										// FSp: 		Pooling Selection (only 2x2 supported)
		0,										// PoolStride:	Pooling Stride (\1, \2)
		0,										// PoolDoPad:	Padded MaxPooling (not yet used)
		0,										// PoolDoReLU:	ReLu after MaxPooling (yes=1, no=0)
		0);										// DoPool:		MaxPooling Enable (yes=1, no=0)


/* --------------------------------- LAYER 9 -------------------------------- */
	/* 3x3 Convolution. Pure SW.
	 * 128 input planes [7x7], 128 output planes [7x7] */
	MediumParOutFeatConvolutionPoolReLU_fp(
		"MedParConv_3x3_S1_9",					// Name: 		Name
		inCh[13],								// InFeat:		Number Input Channels
		outCh[13],								// OutFeat:		Number Output Channels
		inW[13],								// Width:		Feature Map Width
		inH[13],								// Height:		Feature Map Height
		3,										// FSc:			Filter Selection (1x1, 3x3, 5x5)
		1,										// ConvStride:	Convolution Stride (\1, \2) 
		1,										// ConvDoPad:	Padded Convolution (yes=1, no=0)
		0,										// ConvDoReLU:	ReLu after Convolution (yes=1, no=0)
		0,										// FSp: 		Pooling Selection (only 2x2 supported)
		0,										// PoolStride:	Pooling Stride (\1, \2)
		0,										// PoolDoPad:	Padded MaxPooling (not yet used)
		0,										// PoolDoReLU:	ReLu after MaxPooling (yes=1, no=0)
		0);										// DoPool:		MaxPooling Enable (yes=1, no=0)


/* -------------------------------- LAYER 10 -------------------------------- */
	/* 1x1 Convolution Stride 2. Pure SW.
	 * 64 input planes [13x13], 128 output planes [7x7] */
	MediumParOutFeatConvolutionPoolReLU_fp(
		"MedParConv_1x1_S1_ReLU_10",			// Name: 		Name
		inCh[14],								// InFeat:		Number Input Channels
		outCh[14],								// OutFeat:		Number Output Channels
		inW[14],								// Width:		Feature Map Width
		inH[14],								// Height:		Feature Map Height
		1,										// FSc:			Filter Selection (1x1, 3x3, 5x5)
		2,										// ConvStride:	Convolution Stride (\1, \2) 
		0,										// ConvDoPad:	Padded Convolution (yes=1, no=0)
		0,										// ConvDoReLU:	ReLu after Convolution (yes=1, no=0)
		0,										// FSp: 		Pooling Selection (only 2x2 supported)
		0,										// PoolStride:	Pooling Stride (\1, \2)
		0,										// PoolDoPad:	Padded MaxPooling (not yet used)
		0,										// PoolDoReLU:	ReLu after MaxPooling (yes=1, no=0)
		0);										// DoPool:		MaxPooling Enable (yes=1, no=0)


/* -------------------------------- ADD RES 3 ------------------------------- */
	/* Matrix Add, followed by ReLU. Pure SW.
	 * 128 input planes [7x7], 128 output planes [7x7] */		
	CNN_MatrixAddReLu_SW_fp(
		"AddFeatureMapsReLu_SW_3",				// Name: 		Name
		inCh[15],								// InFeat:		Number Input Channels
		outCh[15],								// OutFeat:		Number Output Channels
		inW[15],								// Width:		Feature Map Width
		inH[15]);								// Height:		Feature Map Height


/* --------------------------------- DENSE 1 -------------------------------- */
	/* Linear Layer. Pure SW.
	 * 128 input planes [7x7], Output 1 */
	CNN_TiledLinearLayer(
		"LinearLayer_SW_1",						// Name: 		Name			
		inCh[16],								// InFeat:		Number Input Channels
		outCh[16],								// OutFeat:		Number Output Channels	
		inW[16],								// Width:		Feature Map Width	
		inH[16],								// Height:		Feature Map Height
		1,										// ModeSize:	
		0,										// ReLu:		ReLu after Convolution (yes=1, no=0)
		0);										// CoeffInL3:


/* --------------------------------- DENSE 2 -------------------------------- */
	/* Linear Layer, followed by ReLU. Pure SW.
	 * 128 input planes [7x7], Output 2 */
	CNN_TiledLinearLayer(
		"LinearLayer_SW_2",						// Name: 		Name			
		inCh[17],								// InFeat:		Number Input Channels
		outCh[17],								// OutFeat:		Number Output Channels	
		inW[17],								// Width:		Feature Map Width	
		inH[17],								// Height:		Feature Map Height
		1,										// ModeSize:	
		0,										// ReLu:		ReLu after Convolution (yes=1, no=0)
		0);										// CoeffInL3:
}


int main(int argc, char **argv) {

	// Parse AutoTiler options
	if(TilerParseOptions(argc, argv)) {
		printf("Failed to initialize or incorrect output arguments directory.\n"); 
		return 1;
	}

	// Set AutoTiler configuration
	PULPDronetConfiguration(54400);
	// Load SW CNN basic kernels
	CNN_LoadSoftwareKernelLibrary();
	// Generate PULPDronet
	PULPDronetGenerator();
	// Generate code
	GenerateTilingCode();

	return 0;
}