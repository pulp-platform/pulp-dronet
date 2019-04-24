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
 * File:    PULPDronetKernels.h                                               *
 * Author:  Daniele Palossi <dpalossi@iis.ee.ethz.ch>                         *
 * Date:    10.04.2019                                                        *
 *----------------------------------------------------------------------------*/


#ifndef __PULPDRONETKERNEL_H__
#define __PULPDRONETKERNEL_H__

#include "KernelLibStdTypes.h"
#include "PULPDronetKernelsInit.h"
#include "CNN_BasicKernels.h"
#define _PULP_Dronet_L1_Memory_SIZE 54280
#define _PULP_Dronet_L2_Memory_SIZE 0
extern char *PULP_Dronet_L1_Memory; /* Size given for generation: 54400 bytes, used: 54280 bytes */
extern char *PULP_Dronet_L2_Memory; /* Size used for generation: 0 bytes */
extern void LargeParConv_5x5_S2_Max2x2_S2_H_1(
		short int * __restrict__ In,
		short int * __restrict__ Filter,
		short int * __restrict__ Out,
		unsigned int Norm,
		short int * __restrict__ Bias,
		Kernel_T *Ker);
extern void ReLU_SW_1(
		short int * __restrict__ In,
		short int * __restrict__ Out,
		Kernel_T *Ker);
extern void MedParConv_3x3_S2_ReLU_2(
		short int * __restrict__ In,
		short int * __restrict__ Filter,
		short int * __restrict__ Out,
		unsigned int Norm,
		short int * __restrict__ Bias,
		Kernel_T *Ker);
extern void MedParConv_3x3_S1_3(
		short int * __restrict__ In,
		short int * __restrict__ Filter,
		short int * __restrict__ Out,
		unsigned int Norm,
		short int * __restrict__ Bias,
		Kernel_T *Ker);
extern void MedParConv_1x1_S2_4(
		short int * __restrict__ In,
		short int * __restrict__ Filter,
		short int * __restrict__ Out,
		unsigned int Norm,
		short int * __restrict__ Bias,
		Kernel_T *Ker);
extern void AddFeatureMaps_SW_1(
		short int * __restrict__ In,
		short int * __restrict__ Out,
		Kernel_T *Ker);
extern void ReLU_SW_2(
		short int * __restrict__ In,
		short int * __restrict__ Out,
		Kernel_T *Ker);
extern void MedParConv_3x3_S2_ReLU_5(
		short int * __restrict__ In,
		short int * __restrict__ Filter,
		short int * __restrict__ Out,
		unsigned int Norm,
		short int * __restrict__ Bias,
		Kernel_T *Ker);
extern void MedParConv_3x3_S1_6(
		short int * __restrict__ In,
		short int * __restrict__ Filter,
		short int * __restrict__ Out,
		unsigned int Norm,
		short int * __restrict__ Bias,
		Kernel_T *Ker);
extern void MedParConv_1x1_S2_7(
		short int * __restrict__ In,
		short int * __restrict__ Filter,
		short int * __restrict__ Out,
		unsigned int Norm,
		short int * __restrict__ Bias,
		Kernel_T *Ker);
extern void AddFeatureMaps_SW_2(
		short int * __restrict__ In,
		short int * __restrict__ Out,
		Kernel_T *Ker);
extern void ReLU_SW_3(
		short int * __restrict__ In,
		short int * __restrict__ Out,
		Kernel_T *Ker);
extern void MedParConv_3x3_S2_ReLU_8(
		short int * __restrict__ In,
		short int * __restrict__ Filter,
		short int * __restrict__ Out,
		unsigned int Norm,
		short int * __restrict__ Bias,
		Kernel_T *Ker);
extern void MedParConv_3x3_S1_9(
		short int * __restrict__ In,
		short int * __restrict__ Filter,
		short int * __restrict__ Out,
		unsigned int Norm,
		short int * __restrict__ Bias,
		Kernel_T *Ker);
extern void MedParConv_1x1_S1_ReLU_10(
		short int * __restrict__ In,
		short int * __restrict__ Filter,
		short int * __restrict__ Out,
		unsigned int Norm,
		short int * __restrict__ Bias,
		Kernel_T *Ker);
extern void AddFeatureMapsReLu_SW_3(
		short int * __restrict__ In,
		short int * __restrict__ Out,
		Kernel_T *Ker);
extern void LinearLayer_SW_1(
		Word16 * __restrict__ In,
		Word16 * __restrict__ Filter,
		unsigned int NormFilter,
		Word16 * __restrict__ Bias,
		unsigned int NormBias,
		Word16 * __restrict__ Out,
		int OutSize,
		Kernel_T *Ker);
extern void LinearLayer_SW_2(
		Word16 * __restrict__ In,
		Word16 * __restrict__ Filter,
		unsigned int NormFilter,
		Word16 * __restrict__ Bias,
		unsigned int NormBias,
		Word16 * __restrict__ Out,
		int OutSize,
		Kernel_T *Ker);
#endif
