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
 * File:    PULPDronetKernels.c                                               *
 * Author:  Daniele Palossi <dpalossi@iis.ee.ethz.ch>                         *
 * Date:    10.04.2019                                                        *
 *----------------------------------------------------------------------------*/


#include "PULPDronetKernelsInit.h"
#include "PULPDronetKernels.h"
void LargeParConv_5x5_S2_Max2x2_S2_H_1(
		short int * __restrict__ In,
		short int * __restrict__ Filter,
		short int * __restrict__ Out,
		unsigned int Norm,
		short int * __restrict__ Bias,
		Kernel_T *Ker)

{
	/* Local variables used by this kernel */
	rt_dma_copy_t DmaR_Evt1;
	rt_dma_copy_t DmaR_Evt2;
	rt_dma_copy_t DmaR_Evt3;
	rt_dma_copy_t DmaW_Evt1;
	int Iter;
	int Last, NextLast, NextNextLast;
	int N_Ti = 0;
	int Iter1;
	int Last1, NextLast1, NextNextLast1;
	int N_Ti1 = 0;
	int N_TiIp = 0, InPlane, OutPlane=0;
	KerParSetBias_fp_T S_KerArg0, *KerArg0 = &S_KerArg0;
	KerParConv_fp_T S_KerArg1, *KerArg1 = &S_KerArg1;
	KerParReLUMaxPool_fp_T S_KerArg2, *KerArg2 = &S_KerArg2;

	/* Initialize KerArg, Kernel invariant arguments */
	KerArg0->W = (unsigned short int) (100);
	KerArg0->H = (unsigned short int) (4);
	KerArg0->OutFeatures = (unsigned short int) (32);
	KerArg1->W = (unsigned short int) (200);
	KerArg1->H = (unsigned short int) (11);
	KerArg1->InFeatures = (unsigned short int) (1);
	KerArg1->OutFeatures = (unsigned short int) (32);
	KerArg1->Norm = (Norm);
	KerArg1->NTile = (unsigned short int) 25;
	KerArg1->Orientation = (unsigned short int) (1);
	KerArg1->Pad = (v4s) (16908288);
	KerArg1->TileSize = (unsigned short int) (11);
	KerArg1->TotalSize = (unsigned short int) (204);
	KerArg2->W = (unsigned short int) (100);
	KerArg2->H = (unsigned short int) (4);
	KerArg2->OutFeatures = (unsigned short int) (32);
	KerArg2->Pad = (v4s) (16777216);
	KerArg2->DoReLU = (unsigned short int) (0);
	/* =======================Read First Tile=========================================== */
	/* Initial reads in L2, O_DB or O_BUFF */
	rt_dma_memcpy_2d((rt_pointerT) In+(0), (rt_pointerT) (PULP_Dronet_L1_Memory + 0)+0, (0?3200:3600), 
		80000, (0?3200:3600), RT_DMA_DIR_EXT2LOC, 0, &DmaR_Evt1);
	rt_dma_memcpy((rt_pointerT) Filter+(0), (rt_pointerT) (PULP_Dronet_L1_Memory + 47264)+0, 1600, RT_DMA_DIR_EXT2LOC, 0, &DmaR_Evt2);
	rt_dma_memcpy((rt_pointerT) Bias+0, (rt_pointerT) (PULP_Dronet_L1_Memory + 8800)+0, 64, RT_DMA_DIR_EXT2LOC, 0, &DmaR_Evt3);
	/* Wait for BUFF read in L2 */
	rt_dma_wait(&DmaR_Evt2);
	rt_dma_wait(&DmaR_Evt3);
	/* ===================End Read First Tile=========================================== */
	/* Kernel Iteration Loop on Iter space */
	for (Iter=0; Iter<25; Iter++) {
		/* Loop Iteration Body on Iter space */
		/* Elaborate Last, Next_Last, Next_Next_Last */
		Last = ((Iter+1) == 25); NextLast = ((Iter+2) == 25); NextNextLast = ((Iter+3) == 25);
		/* =======================Read Next Tile=========================================== */
		rt_dma_wait(&DmaR_Evt1);
		if (!Last) {
			rt_dma_memcpy_2d((rt_pointerT) In + ((Iter+1)*3200-800),
					(rt_pointerT) (PULP_Dronet_L1_Memory + 0) + (4400*((N_Ti+1) % 2)), (NextLast?4000:4400), 
					80000, (NextLast?4000:4400), RT_DMA_DIR_EXT2LOC, 0, &DmaR_Evt1);
		}
		/* ===================End Read Next Tile=========================================== */
		/* Call Kernel LOC_Unknown */
		KerArg0->Out = (short int * __restrict__) ((rt_pointerT) (PULP_Dronet_L1_Memory + 21664) + 0);
		KerArg0->Bias = (short int * __restrict__) ((rt_pointerT) (PULP_Dronet_L1_Memory + 8800) + 0);
		rt_team_fork(gap8_ncore(), (void *) KerParSetBias_fp, (void *) KerArg0);
		/* Kernel Iteration Loop on Iter1 space */
		Iter1=0; {
			/* Loop Iteration Body on Iter1 space */
			/* Elaborate Last, Next_Last, Next_Next_Last */
			Last1 = ((Iter1+1) == 1); NextLast1 = ((Iter1+2) == 1); NextNextLast1 = ((Iter1+3) == 1);
			/* Call Kernel LOC_Unknown */
			KerArg1->In = (short int * __restrict__) ((rt_pointerT) (PULP_Dronet_L1_Memory + 0) + 4400*((N_Ti) % 2));
			KerArg1->BaseOutFeature = (unsigned short int) (0)*32;
			KerArg1->Filter = (short int * __restrict__) ((rt_pointerT) (PULP_Dronet_L1_Memory + 47264) + 0 + (0)*1600);
			KerArg1->Out = (short int * __restrict__) ((rt_pointerT) (PULP_Dronet_L1_Memory + 21664) + 0);
			KerArg1->TileIndex = (unsigned short int) Iter;
			rt_team_fork(gap8_ncore(), (void *) KerParConv5x5Stride2_fp, (void *) KerArg1);
			N_Ti1++;
			/* End Kernel Iteration Loop on Iter1 space */
		}
		Iter1=1;
		/* Call Kernel LOC_Unknown */
		KerArg2->In = (short int * __restrict__) ((rt_pointerT) (PULP_Dronet_L1_Memory + 21664) + 0);
		KerArg2->Out = (short int * __restrict__) ((rt_pointerT) (PULP_Dronet_L1_Memory + 8864) + 6400*((N_Ti) % 2));
		rt_team_fork(gap8_ncore(), (void *) KerParMaxPool2x2Stride2_fp, (void *) KerArg2);
		/* =======================Write Tile=========================================== */
		if (Iter) {
			rt_dma_wait(&DmaW_Evt1);
		}
		rt_dma_memcpy_2d((rt_pointerT) Out + ((Iter)*200),
			(rt_pointerT) (PULP_Dronet_L1_Memory + 8864) + (6400*(N_Ti % 2)), 6400, 
			5000, 200, RT_DMA_DIR_LOC2EXT, 0, &DmaW_Evt1);
		/* ===================End Write Tile=========================================== */
		N_Ti++;
		/* End Kernel Iteration Loop on Iter space */
	}
	Iter=25;
	/* =======================Write Last Tile=========================================== */
	rt_dma_wait(&DmaW_Evt1);
	/* ===================End Write Last Tile=========================================== */
}

void ReLU_SW_1(
		short int * __restrict__ In,
		short int * __restrict__ Out,
		Kernel_T *Ker)

{
	/* Local variables used by this kernel */
	rt_dma_copy_t DmaR_Evt1;
	rt_dma_copy_t DmaR_Evt2;
	rt_dma_copy_t DmaW_Evt1;
	int Iter;
	int Last, NextLast, NextNextLast;
	int N_Ti = 0;
	int N_TiIp = 0, InPlane, OutPlane=0;
	KerReLUMaxPool2x2_fpT S_KerArg0, *KerArg0 = &S_KerArg0;

	/* Initialize KerArg, Kernel invariant arguments */
	KerArg0->W = (int) (50);
	KerArg0->H = (int) (50);
	/* =======================Read First Tile=========================================== */
	/* Initial reads in L2, O_DB or O_BUFF */
	rt_dma_memcpy((rt_pointerT) In+(0), (rt_pointerT) (PULP_Dronet_L1_Memory + 0)+0, 5000, RT_DMA_DIR_EXT2LOC, 0, &DmaR_Evt1);
	rt_dma_memcpy((rt_pointerT) Out+(0), (rt_pointerT) (PULP_Dronet_L1_Memory + 10000)+0, 5000, RT_DMA_DIR_EXT2LOC, 0, &DmaR_Evt2);
	/* ===================End Read First Tile=========================================== */
	/* Kernel Iteration Loop on Input+Output Planes */
	for (OutPlane=0; OutPlane<32; OutPlane++) {
		int LastOutPlane = ((OutPlane+1) == 32), NextLastOutPlane = ((OutPlane+2) == 32);
		/* Kernel Iteration Loop on Iter space */
		Iter=0; {
			/* Loop Iteration Body on Iter space */
			/* Elaborate Last, Next_Last, Next_Next_Last */
			Last = ((Iter+1) == 1); NextLast = ((Iter+2) == 1); NextNextLast = ((Iter+3) == 1);
			/* =======================Read Next Tile=========================================== */
			rt_dma_wait(&DmaR_Evt1);
			rt_dma_wait(&DmaR_Evt2);
			if (!Last) {
				rt_dma_memcpy((rt_pointerT) In + ((0+1)*5000),
						(rt_pointerT) (PULP_Dronet_L1_Memory + 0) + (5000*((N_Ti+1) % 2)), 5000, RT_DMA_DIR_EXT2LOC, 0, &DmaR_Evt1);
				rt_dma_memcpy((rt_pointerT) Out + ((0+1)*5000),
						(rt_pointerT) (PULP_Dronet_L1_Memory + 10000) + (5000*((N_Ti+1) % 3)), 5000, RT_DMA_DIR_EXT2LOC, 0, &DmaR_Evt2);
			}
			else if (!LastOutPlane) {
				rt_dma_memcpy((rt_pointerT) In + ((1)*5000 + 0),
						(rt_pointerT) (PULP_Dronet_L1_Memory + 0) + (5000*((N_Ti+1) % 2)), 5000, RT_DMA_DIR_EXT2LOC, 0, &DmaR_Evt1);
				rt_dma_memcpy((rt_pointerT) Out + ((1)*5000 + 0),
						(rt_pointerT) (PULP_Dronet_L1_Memory + 10000) + (5000*((N_Ti+1) % 3)), 5000, RT_DMA_DIR_EXT2LOC, 0, &DmaR_Evt2);
			}
			/* ===================End Read Next Tile=========================================== */
			/* Call Kernel LOC_INNER_LOOP */
			KerArg0->In = (short int * __restrict__) ((rt_pointerT) (PULP_Dronet_L1_Memory + 0) + 5000*((N_Ti) % 2));
			KerArg0->Out = (short int * __restrict__) ((rt_pointerT) (PULP_Dronet_L1_Memory + 10000) + 5000*((N_Ti) % 3));
			rt_team_fork(gap8_ncore(), (void *) KerReLU_fp, (void *) KerArg0);
			/* =======================Write Tile=========================================== */
			if (0||OutPlane) {
				rt_dma_wait(&DmaW_Evt1);
			}
			rt_dma_memcpy((rt_pointerT) Out + ((0)*5000),
				(rt_pointerT) (PULP_Dronet_L1_Memory + 10000) + (5000*(N_Ti % 3)), 5000, RT_DMA_DIR_LOC2EXT, 0, &DmaW_Evt1);
			/* ===================End Write Tile=========================================== */
			N_Ti++;
			/* End Kernel Iteration Loop on Iter space */
		}
		Iter=1;
		In = In + 2500;
		Out = Out + 2500;
		/* End Kernel Iteration Loop on Input+Output Planes */
	}
	/* =======================Write Last Tile=========================================== */
	rt_dma_wait(&DmaW_Evt1);
	/* ===================End Write Last Tile=========================================== */
}

void MedParConv_3x3_S2_ReLU_2(
		short int * __restrict__ In,
		short int * __restrict__ Filter,
		short int * __restrict__ Out,
		unsigned int Norm,
		short int * __restrict__ Bias,
		Kernel_T *Ker)

{
	/* Local variables used by this kernel */
	rt_dma_copy_t DmaR_Evt1;
	rt_dma_copy_t DmaR_Evt2;
	rt_dma_copy_t DmaR_Evt3;
	rt_dma_copy_t DmaW_Evt1;
	int Iter;
	int Last, NextLast, NextNextLast;
	int N_Ti = 0;
	int Iter1;
	int Last1, NextLast1, NextNextLast1;
	int N_Ti1 = 0;
	int N_TiIp = 0, InPlane, OutPlane=0;
	KerParSetBias_fp_T S_KerArg0, *KerArg0 = &S_KerArg0;
	KerParConv_fp_T S_KerArg1, *KerArg1 = &S_KerArg1;
	KerParReLUMaxPool_fp_T S_KerArg2, *KerArg2 = &S_KerArg2;

	/* Initialize KerArg, Kernel invariant arguments */
	KerArg0->W = (unsigned short int) (25);
	KerArg0->H = (unsigned short int) (25);
	KerArg0->OutFeatures = (unsigned short int) (8);
	KerArg1->W = (unsigned short int) (50);
	KerArg1->H = (unsigned short int) (50);
	KerArg1->InFeatures = (unsigned short int) (2);
	KerArg1->OutFeatures = (unsigned short int) (8);
	KerArg1->Norm = (Norm);
	KerArg1->TileIndex = (unsigned short int) (32);
	KerArg1->NTile = (unsigned short int) (0);
	KerArg1->Pad = (v4s) (16843009);
	KerArg2->W = (unsigned short int) (25);
	KerArg2->H = (unsigned short int) (25);
	KerArg2->OutFeatures = (unsigned short int) (8);
	/* =======================Read First Tile=========================================== */
	/* Initial reads in L2, O_DB or O_BUFF */
	rt_dma_memcpy((rt_pointerT) In+(0), (rt_pointerT) (PULP_Dronet_L1_Memory + 29280)+0, 10000, RT_DMA_DIR_EXT2LOC, 0, &DmaR_Evt1);
	rt_dma_memcpy((rt_pointerT) Filter+(0), (rt_pointerT) (PULP_Dronet_L1_Memory + 0)+0, 4608, RT_DMA_DIR_EXT2LOC, 0, &DmaR_Evt2);
	rt_dma_memcpy((rt_pointerT) Bias+(0), (rt_pointerT) (PULP_Dronet_L1_Memory + 9216)+0, 64, RT_DMA_DIR_EXT2LOC, 0, &DmaR_Evt3);
	/* Wait for BUFF read in L2 */
	rt_dma_wait(&DmaR_Evt3);
	/* ===================End Read First Tile=========================================== */
	/* Kernel Iteration Loop on Iter space */
	for (Iter=0; Iter<4; Iter++) {
		/* Loop Iteration Body on Iter space */
		/* Elaborate Last, Next_Last, Next_Next_Last */
		Last = ((Iter+1) == 4); NextLast = ((Iter+2) == 4); NextNextLast = ((Iter+3) == 4);
		/* =======================Read Next Tile=========================================== */
		rt_dma_wait(&DmaR_Evt2);
		if (!Last) {
			rt_dma_memcpy((rt_pointerT) Filter + ((Iter+1)*4608),
					(rt_pointerT) (PULP_Dronet_L1_Memory + 0) + (4608*((N_Ti+1) % 2)), 4608, RT_DMA_DIR_EXT2LOC, 0, &DmaR_Evt2);
		}
		/* ===================End Read Next Tile=========================================== */
		/* Call Kernel LOC_Unknown */
		KerArg0->Out = (short int * __restrict__) ((rt_pointerT) (PULP_Dronet_L1_Memory + 9280) + 10000*((N_Ti) % 2));
		KerArg0->Bias = (short int * __restrict__) ((rt_pointerT) (PULP_Dronet_L1_Memory + 9216) + 0 + (Iter)*16);
		rt_team_fork(gap8_ncore(), (void *) KerParSetBias_fp, (void *) KerArg0);
		/* Kernel Iteration Loop on Iter1 space */
		for (Iter1=0; Iter1<16; Iter1++) {
			/* Loop Iteration Body on Iter1 space */
			/* Elaborate Last, Next_Last, Next_Next_Last */
			Last1 = ((Iter1+1) == 16); NextLast1 = ((Iter1+2) == 16); NextNextLast1 = ((Iter1+3) == 16);
			/* =======================Read Next Tile=========================================== */
			rt_dma_wait(&DmaR_Evt1);
			if (!Last1) {
				rt_dma_memcpy((rt_pointerT) In + ((Iter1+1)*10000),
						(rt_pointerT) (PULP_Dronet_L1_Memory + 29280) + (10000*((N_Ti1+1) % 2)), 10000, RT_DMA_DIR_EXT2LOC, 0, &DmaR_Evt1);
			}
			else if (!Last) {
				rt_dma_memcpy((rt_pointerT) In + (0),
						(rt_pointerT) (PULP_Dronet_L1_Memory + 29280) + (10000*((N_Ti1+1) % 2)), 10000, RT_DMA_DIR_EXT2LOC, 0, &DmaR_Evt1);
			}
			/* ===================End Read Next Tile=========================================== */
			/* Call Kernel LOC_Unknown */
			KerArg1->In = (short int * __restrict__) ((rt_pointerT) (PULP_Dronet_L1_Memory + 29280) + 10000*((N_Ti1) % 2));
			KerArg1->BaseOutFeature = (unsigned short int) (Iter1)*2;
			KerArg1->Filter = (short int * __restrict__) ((rt_pointerT) (PULP_Dronet_L1_Memory + 0) + 4608*((N_Ti) % 2));
			KerArg1->Out = (short int * __restrict__) ((rt_pointerT) (PULP_Dronet_L1_Memory + 9280) + 10000*((N_Ti) % 2));
			rt_team_fork(gap8_ncore(), (void *) KerParConv3x3Stride2_fp, (void *) KerArg1);
			N_Ti1++;
			/* End Kernel Iteration Loop on Iter1 space */
		}
		Iter1=16;
		/* Call Kernel LOC_Unknown */
		KerArg2->In = (short int * __restrict__) ((rt_pointerT) (PULP_Dronet_L1_Memory + 9280) + 10000*((N_Ti) % 2));
		KerArg2->Out = (short int * __restrict__) ((rt_pointerT) (PULP_Dronet_L1_Memory + 9280) + 10000*((N_Ti) % 2));
		rt_team_fork(gap8_ncore(), (void *) KerParReLU_fp, (void *) KerArg2);
		/* =======================Write Tile=========================================== */
		if (Iter) {
			rt_dma_wait(&DmaW_Evt1);
		}
		rt_dma_memcpy((rt_pointerT) Out + ((Iter)*10000),
			(rt_pointerT) (PULP_Dronet_L1_Memory + 9280) + (10000*(N_Ti % 2)), 10000, RT_DMA_DIR_LOC2EXT, 0, &DmaW_Evt1);
		/* ===================End Write Tile=========================================== */
		N_Ti++;
		/* End Kernel Iteration Loop on Iter space */
	}
	Iter=4;
	/* =======================Write Last Tile=========================================== */
	rt_dma_wait(&DmaW_Evt1);
	/* ===================End Write Last Tile=========================================== */
}

void MedParConv_3x3_S1_3(
		short int * __restrict__ In,
		short int * __restrict__ Filter,
		short int * __restrict__ Out,
		unsigned int Norm,
		short int * __restrict__ Bias,
		Kernel_T *Ker)

{
	/* Local variables used by this kernel */
	rt_dma_copy_t DmaR_Evt1;
	rt_dma_copy_t DmaR_Evt2;
	rt_dma_copy_t DmaR_Evt3;
	rt_dma_copy_t DmaW_Evt1;
	int Iter;
	int Last, NextLast, NextNextLast;
	int N_Ti = 0;
	int Iter1;
	int Last1, NextLast1, NextNextLast1;
	int N_Ti1 = 0;
	int N_TiIp = 0, InPlane, OutPlane=0;
	KerParSetBias_fp_T S_KerArg0, *KerArg0 = &S_KerArg0;
	KerParConv_fp_T S_KerArg1, *KerArg1 = &S_KerArg1;

	/* Initialize KerArg, Kernel invariant arguments */
	KerArg0->W = (unsigned short int) (25);
	KerArg0->H = (unsigned short int) (25);
	KerArg0->OutFeatures = (unsigned short int) (8);
	KerArg1->W = (unsigned short int) (25);
	KerArg1->H = (unsigned short int) (25);
	KerArg1->OutFeatures = (unsigned short int) (8);
	KerArg1->Norm = (Norm);
	KerArg1->TileIndex = (unsigned short int) (32);
	KerArg1->NTile = (unsigned short int) (0);
	KerArg1->Pad = (v4s) (16843009);
	/* =======================Read First Tile=========================================== */
	/* Initial reads in L2, O_DB or O_BUFF */
	rt_dma_memcpy((rt_pointerT) In+(0), (rt_pointerT) (PULP_Dronet_L1_Memory + 29280)+0, 12500, RT_DMA_DIR_EXT2LOC, 0, &DmaR_Evt1);
	rt_dma_memcpy((rt_pointerT) Filter+(0), (rt_pointerT) (PULP_Dronet_L1_Memory + 0)+0, 4608, RT_DMA_DIR_EXT2LOC, 0, &DmaR_Evt2);
	rt_dma_memcpy((rt_pointerT) Bias+(0), (rt_pointerT) (PULP_Dronet_L1_Memory + 9216)+0, 64, RT_DMA_DIR_EXT2LOC, 0, &DmaR_Evt3);
	/* Wait for BUFF read in L2 */
	rt_dma_wait(&DmaR_Evt3);
	/* ===================End Read First Tile=========================================== */
	/* Kernel Iteration Loop on Iter space */
	for (Iter=0; Iter<4; Iter++) {
		/* Loop Iteration Body on Iter space */
		/* Elaborate Last, Next_Last, Next_Next_Last */
		Last = ((Iter+1) == 4); NextLast = ((Iter+2) == 4); NextNextLast = ((Iter+3) == 4);
		/* =======================Read Next Tile=========================================== */
		rt_dma_wait(&DmaR_Evt2);
		if (!Last) {
			rt_dma_memcpy((rt_pointerT) Filter + ((Iter+1)*4608),
					(rt_pointerT) (PULP_Dronet_L1_Memory + 0) + (4608*((N_Ti+1) % 2)), 4608, RT_DMA_DIR_EXT2LOC, 0, &DmaR_Evt2);
		}
		/* ===================End Read Next Tile=========================================== */
		/* Call Kernel LOC_Unknown */
		KerArg0->Out = (short int * __restrict__) ((rt_pointerT) (PULP_Dronet_L1_Memory + 9280) + 10000*((N_Ti) % 2));
		KerArg0->Bias = (short int * __restrict__) ((rt_pointerT) (PULP_Dronet_L1_Memory + 9216) + 0 + (Iter)*16);
		rt_team_fork(gap8_ncore(), (void *) KerParSetBias_fp, (void *) KerArg0);
		/* Kernel Iteration Loop on Iter1 space */
		for (Iter1=0; Iter1<4; Iter1++) {
			/* Loop Iteration Body on Iter1 space */
			/* Elaborate Last, Next_Last, Next_Next_Last */
			Last1 = ((Iter1+1) == 4); NextLast1 = ((Iter1+2) == 4); NextNextLast1 = ((Iter1+3) == 4);
			/* =======================Read Next Tile=========================================== */
			rt_dma_wait(&DmaR_Evt1);
			if (!Last1) {
				rt_dma_memcpy((rt_pointerT) In + ((Iter1+1)*12500),
						(rt_pointerT) (PULP_Dronet_L1_Memory + 29280) + (12500*((N_Ti1+1) % 2)), NextLast1?2500:12500, RT_DMA_DIR_EXT2LOC, 0, &DmaR_Evt1);
			}
			else if (!Last) {
				rt_dma_memcpy((rt_pointerT) In + (0),
						(rt_pointerT) (PULP_Dronet_L1_Memory + 29280) + (12500*((N_Ti1+1) % 2)), 12500, RT_DMA_DIR_EXT2LOC, 0, &DmaR_Evt1);
			}
			/* ===================End Read Next Tile=========================================== */
			/* Call Kernel LOC_Unknown */
			KerArg1->In = (short int * __restrict__) ((rt_pointerT) (PULP_Dronet_L1_Memory + 29280) + 12500*((N_Ti1) % 2));
			KerArg1->InFeatures = (unsigned short int) (Last1?2:10);
			KerArg1->BaseOutFeature = (unsigned short int) (Iter1)*10;
			KerArg1->Filter = (short int * __restrict__) ((rt_pointerT) (PULP_Dronet_L1_Memory + 0) + 4608*((N_Ti) % 2));
			KerArg1->Out = (short int * __restrict__) ((rt_pointerT) (PULP_Dronet_L1_Memory + 9280) + 10000*((N_Ti) % 2));
			rt_team_fork(gap8_ncore(), (void *) KerParConv3x3Stride1_fp, (void *) KerArg1);
			N_Ti1++;
			/* End Kernel Iteration Loop on Iter1 space */
		}
		Iter1=4;
		/* =======================Write Tile=========================================== */
		if (Iter) {
			rt_dma_wait(&DmaW_Evt1);
		}
		rt_dma_memcpy((rt_pointerT) Out + ((Iter)*10000),
			(rt_pointerT) (PULP_Dronet_L1_Memory + 9280) + (10000*(N_Ti % 2)), 10000, RT_DMA_DIR_LOC2EXT, 0, &DmaW_Evt1);
		/* ===================End Write Tile=========================================== */
		N_Ti++;
		/* End Kernel Iteration Loop on Iter space */
	}
	Iter=4;
	/* =======================Write Last Tile=========================================== */
	rt_dma_wait(&DmaW_Evt1);
	/* ===================End Write Last Tile=========================================== */
}

void MedParConv_1x1_S2_4(
		short int * __restrict__ In,
		short int * __restrict__ Filter,
		short int * __restrict__ Out,
		unsigned int Norm,
		short int * __restrict__ Bias,
		Kernel_T *Ker)

{
	/* Local variables used by this kernel */
	rt_dma_copy_t DmaR_Evt1;
	rt_dma_copy_t DmaR_Evt2;
	rt_dma_copy_t DmaR_Evt3;
	rt_dma_copy_t DmaW_Evt1;
	int Iter;
	int Last, NextLast, NextNextLast;
	int N_Ti = 0;
	int Iter1;
	int Last1, NextLast1, NextNextLast1;
	int N_Ti1 = 0;
	int N_TiIp = 0, InPlane, OutPlane=0;
	KerParSetBias_fp_T S_KerArg0, *KerArg0 = &S_KerArg0;
	KerParConv_fp_T S_KerArg1, *KerArg1 = &S_KerArg1;

	/* Initialize KerArg, Kernel invariant arguments */
	KerArg0->W = (unsigned short int) (25);
	KerArg0->H = (unsigned short int) (25);
	KerArg0->OutFeatures = (unsigned short int) (8);
	KerArg1->W = (unsigned short int) (50);
	KerArg1->H = (unsigned short int) (50);
	KerArg1->OutFeatures = (unsigned short int) (8);
	KerArg1->Norm = (Norm);
	KerArg1->TileIndex = (unsigned short int) (32);
	KerArg1->NTile = (unsigned short int) (0);
	KerArg1->Pad = (v4s) (0);
	/* =======================Read First Tile=========================================== */
	/* Initial reads in L2, O_DB or O_BUFF */
	rt_dma_memcpy((rt_pointerT) In+(0), (rt_pointerT) (PULP_Dronet_L1_Memory + 21088)+0, 15000, RT_DMA_DIR_EXT2LOC, 0, &DmaR_Evt1);
	rt_dma_memcpy((rt_pointerT) Filter+(0), (rt_pointerT) (PULP_Dronet_L1_Memory + 0)+0, 512, RT_DMA_DIR_EXT2LOC, 0, &DmaR_Evt2);
	rt_dma_memcpy((rt_pointerT) Bias+(0), (rt_pointerT) (PULP_Dronet_L1_Memory + 1024)+0, 64, RT_DMA_DIR_EXT2LOC, 0, &DmaR_Evt3);
	/* Wait for BUFF read in L2 */
	rt_dma_wait(&DmaR_Evt3);
	/* ===================End Read First Tile=========================================== */
	/* Kernel Iteration Loop on Iter space */
	for (Iter=0; Iter<4; Iter++) {
		/* Loop Iteration Body on Iter space */
		/* Elaborate Last, Next_Last, Next_Next_Last */
		Last = ((Iter+1) == 4); NextLast = ((Iter+2) == 4); NextNextLast = ((Iter+3) == 4);
		/* =======================Read Next Tile=========================================== */
		rt_dma_wait(&DmaR_Evt2);
		if (!Last) {
			rt_dma_memcpy((rt_pointerT) Filter + ((Iter+1)*512),
					(rt_pointerT) (PULP_Dronet_L1_Memory + 0) + (512*((N_Ti+1) % 2)), 512, RT_DMA_DIR_EXT2LOC, 0, &DmaR_Evt2);
		}
		/* ===================End Read Next Tile=========================================== */
		/* Call Kernel LOC_Unknown */
		KerArg0->Out = (short int * __restrict__) ((rt_pointerT) (PULP_Dronet_L1_Memory + 1088) + 10000*((N_Ti) % 2));
		KerArg0->Bias = (short int * __restrict__) ((rt_pointerT) (PULP_Dronet_L1_Memory + 1024) + 0 + (Iter)*16);
		rt_team_fork(gap8_ncore(), (void *) KerParSetBias_fp, (void *) KerArg0);
		/* Kernel Iteration Loop on Iter1 space */
		for (Iter1=0; Iter1<11; Iter1++) {
			/* Loop Iteration Body on Iter1 space */
			/* Elaborate Last, Next_Last, Next_Next_Last */
			Last1 = ((Iter1+1) == 11); NextLast1 = ((Iter1+2) == 11); NextNextLast1 = ((Iter1+3) == 11);
			/* =======================Read Next Tile=========================================== */
			rt_dma_wait(&DmaR_Evt1);
			if (!Last1) {
				rt_dma_memcpy((rt_pointerT) In + ((Iter1+1)*15000),
						(rt_pointerT) (PULP_Dronet_L1_Memory + 21088) + (15000*((N_Ti1+1) % 2)), NextLast1?10000:15000, RT_DMA_DIR_EXT2LOC, 0, &DmaR_Evt1);
			}
			else if (!Last) {
				rt_dma_memcpy((rt_pointerT) In + (0),
						(rt_pointerT) (PULP_Dronet_L1_Memory + 21088) + (15000*((N_Ti1+1) % 2)), 15000, RT_DMA_DIR_EXT2LOC, 0, &DmaR_Evt1);
			}
			/* ===================End Read Next Tile=========================================== */
			/* Call Kernel LOC_Unknown */
			KerArg1->In = (short int * __restrict__) ((rt_pointerT) (PULP_Dronet_L1_Memory + 21088) + 15000*((N_Ti1) % 2));
			KerArg1->InFeatures = (unsigned short int) (Last1?2:3);
			KerArg1->BaseOutFeature = (unsigned short int) (Iter1)*3;
			KerArg1->Filter = (short int * __restrict__) ((rt_pointerT) (PULP_Dronet_L1_Memory + 0) + 512*((N_Ti) % 2));
			KerArg1->Out = (short int * __restrict__) ((rt_pointerT) (PULP_Dronet_L1_Memory + 1088) + 10000*((N_Ti) % 2));
			rt_team_fork(gap8_ncore(), (void *) KerParConv1x1Stride2_fp, (void *) KerArg1);
			N_Ti1++;
			/* End Kernel Iteration Loop on Iter1 space */
		}
		Iter1=11;
		/* =======================Write Tile=========================================== */
		if (Iter) {
			rt_dma_wait(&DmaW_Evt1);
		}
		rt_dma_memcpy((rt_pointerT) Out + ((Iter)*10000),
			(rt_pointerT) (PULP_Dronet_L1_Memory + 1088) + (10000*(N_Ti % 2)), 10000, RT_DMA_DIR_LOC2EXT, 0, &DmaW_Evt1);
		/* ===================End Write Tile=========================================== */
		N_Ti++;
		/* End Kernel Iteration Loop on Iter space */
	}
	Iter=4;
	/* =======================Write Last Tile=========================================== */
	rt_dma_wait(&DmaW_Evt1);
	/* ===================End Write Last Tile=========================================== */
}

void AddFeatureMaps_SW_1(
		short int * __restrict__ In,
		short int * __restrict__ Out,
		Kernel_T *Ker)

{
	/* Local variables used by this kernel */
	rt_dma_copy_t DmaR_Evt1;
	rt_dma_copy_t DmaR_Evt2;
	rt_dma_copy_t DmaW_Evt1;
	int Iter;
	int Last, NextLast, NextNextLast;
	int N_Ti = 0;
	int N_TiIp = 0, InPlane, OutPlane=0;
	KerAddFM_fpT S_KerArg0, *KerArg0 = &S_KerArg0;

	/* Initialize KerArg, Kernel invariant arguments */
	KerArg0->W = (int) (25);
	KerArg0->H = (int) (25);
	/* =======================Read First Tile=========================================== */
	/* Initial reads in L2, O_DB or O_BUFF */
	rt_dma_memcpy((rt_pointerT) In+(0), (rt_pointerT) (PULP_Dronet_L1_Memory + 0)+0, 1250, RT_DMA_DIR_EXT2LOC, 0, &DmaR_Evt1);
	rt_dma_memcpy((rt_pointerT) Out+(0), (rt_pointerT) (PULP_Dronet_L1_Memory + 2504)+0, 1250, RT_DMA_DIR_EXT2LOC, 0, &DmaR_Evt2);
	/* ===================End Read First Tile=========================================== */
	/* Kernel Iteration Loop on Input+Output Planes */
	for (OutPlane=0; OutPlane<32; OutPlane++) {
		int LastOutPlane = ((OutPlane+1) == 32), NextLastOutPlane = ((OutPlane+2) == 32);
		/* Kernel Iteration Loop on Iter space */
		Iter=0; {
			/* Loop Iteration Body on Iter space */
			/* Elaborate Last, Next_Last, Next_Next_Last */
			Last = ((Iter+1) == 1); NextLast = ((Iter+2) == 1); NextNextLast = ((Iter+3) == 1);
			/* =======================Read Next Tile=========================================== */
			rt_dma_wait(&DmaR_Evt1);
			rt_dma_wait(&DmaR_Evt2);
			if (!Last) {
				rt_dma_memcpy((rt_pointerT) In + ((0+1)*1250),
						(rt_pointerT) (PULP_Dronet_L1_Memory + 0) + (1252*((N_Ti+1) % 2)), 1250, RT_DMA_DIR_EXT2LOC, 0, &DmaR_Evt1);
				rt_dma_memcpy((rt_pointerT) Out + ((0+1)*1250),
						(rt_pointerT) (PULP_Dronet_L1_Memory + 2504) + (1252*((N_Ti+1) % 3)), 1250, RT_DMA_DIR_EXT2LOC, 0, &DmaR_Evt2);
			}
			else if (!LastOutPlane) {
				rt_dma_memcpy((rt_pointerT) In + ((1)*1250 + 0),
						(rt_pointerT) (PULP_Dronet_L1_Memory + 0) + (1252*((N_Ti+1) % 2)), 1250, RT_DMA_DIR_EXT2LOC, 0, &DmaR_Evt1);
				rt_dma_memcpy((rt_pointerT) Out + ((1)*1250 + 0),
						(rt_pointerT) (PULP_Dronet_L1_Memory + 2504) + (1252*((N_Ti+1) % 3)), 1250, RT_DMA_DIR_EXT2LOC, 0, &DmaR_Evt2);
			}
			/* ===================End Read Next Tile=========================================== */
			/* Call Kernel LOC_INNER_LOOP */
			KerArg0->In = (Word16 * __restrict__) ((rt_pointerT) (PULP_Dronet_L1_Memory + 0) + 1252*((N_Ti) % 2));
			KerArg0->Out = (Word16 * __restrict__) ((rt_pointerT) (PULP_Dronet_L1_Memory + 2504) + 1252*((N_Ti) % 3));
			rt_team_fork(gap8_ncore(), (void *) KerAddFM_fp, (void *) KerArg0);
			/* =======================Write Tile=========================================== */
			if (0||OutPlane) {
				rt_dma_wait(&DmaW_Evt1);
			}
			rt_dma_memcpy((rt_pointerT) Out + ((0)*1250),
				(rt_pointerT) (PULP_Dronet_L1_Memory + 2504) + (1252*(N_Ti % 3)), 1250, RT_DMA_DIR_LOC2EXT, 0, &DmaW_Evt1);
			/* ===================End Write Tile=========================================== */
			N_Ti++;
			/* End Kernel Iteration Loop on Iter space */
		}
		Iter=1;
		In = In + 625;
		Out = Out + 625;
		/* End Kernel Iteration Loop on Input+Output Planes */
	}
	/* =======================Write Last Tile=========================================== */
	rt_dma_wait(&DmaW_Evt1);
	/* ===================End Write Last Tile=========================================== */
}

void ReLU_SW_2(
		short int * __restrict__ In,
		short int * __restrict__ Out,
		Kernel_T *Ker)

{
	/* Local variables used by this kernel */
	rt_dma_copy_t DmaR_Evt1;
	rt_dma_copy_t DmaR_Evt2;
	rt_dma_copy_t DmaW_Evt1;
	int Iter;
	int Last, NextLast, NextNextLast;
	int N_Ti = 0;
	int N_TiIp = 0, InPlane, OutPlane=0;
	KerReLUMaxPool2x2_fpT S_KerArg0, *KerArg0 = &S_KerArg0;

	/* Initialize KerArg, Kernel invariant arguments */
	KerArg0->W = (int) (25);
	KerArg0->H = (int) (25);
	/* =======================Read First Tile=========================================== */
	/* Initial reads in L2, O_DB or O_BUFF */
	rt_dma_memcpy((rt_pointerT) In+(0), (rt_pointerT) (PULP_Dronet_L1_Memory + 0)+0, 1250, RT_DMA_DIR_EXT2LOC, 0, &DmaR_Evt1);
	rt_dma_memcpy((rt_pointerT) Out+(0), (rt_pointerT) (PULP_Dronet_L1_Memory + 2504)+0, 1250, RT_DMA_DIR_EXT2LOC, 0, &DmaR_Evt2);
	/* ===================End Read First Tile=========================================== */
	/* Kernel Iteration Loop on Input+Output Planes */
	for (OutPlane=0; OutPlane<32; OutPlane++) {
		int LastOutPlane = ((OutPlane+1) == 32), NextLastOutPlane = ((OutPlane+2) == 32);
		/* Kernel Iteration Loop on Iter space */
		Iter=0; {
			/* Loop Iteration Body on Iter space */
			/* Elaborate Last, Next_Last, Next_Next_Last */
			Last = ((Iter+1) == 1); NextLast = ((Iter+2) == 1); NextNextLast = ((Iter+3) == 1);
			/* =======================Read Next Tile=========================================== */
			rt_dma_wait(&DmaR_Evt1);
			rt_dma_wait(&DmaR_Evt2);
			if (!Last) {
				rt_dma_memcpy((rt_pointerT) In + ((0+1)*1250),
						(rt_pointerT) (PULP_Dronet_L1_Memory + 0) + (1252*((N_Ti+1) % 2)), 1250, RT_DMA_DIR_EXT2LOC, 0, &DmaR_Evt1);
				rt_dma_memcpy((rt_pointerT) Out + ((0+1)*1250),
						(rt_pointerT) (PULP_Dronet_L1_Memory + 2504) + (1252*((N_Ti+1) % 3)), 1250, RT_DMA_DIR_EXT2LOC, 0, &DmaR_Evt2);
			}
			else if (!LastOutPlane) {
				rt_dma_memcpy((rt_pointerT) In + ((1)*1250 + 0),
						(rt_pointerT) (PULP_Dronet_L1_Memory + 0) + (1252*((N_Ti+1) % 2)), 1250, RT_DMA_DIR_EXT2LOC, 0, &DmaR_Evt1);
				rt_dma_memcpy((rt_pointerT) Out + ((1)*1250 + 0),
						(rt_pointerT) (PULP_Dronet_L1_Memory + 2504) + (1252*((N_Ti+1) % 3)), 1250, RT_DMA_DIR_EXT2LOC, 0, &DmaR_Evt2);
			}
			/* ===================End Read Next Tile=========================================== */
			/* Call Kernel LOC_INNER_LOOP */
			KerArg0->In = (short int * __restrict__) ((rt_pointerT) (PULP_Dronet_L1_Memory + 0) + 1252*((N_Ti) % 2));
			KerArg0->Out = (short int * __restrict__) ((rt_pointerT) (PULP_Dronet_L1_Memory + 2504) + 1252*((N_Ti) % 3));
			rt_team_fork(gap8_ncore(), (void *) KerReLU_fp, (void *) KerArg0);
			/* =======================Write Tile=========================================== */
			if (0||OutPlane) {
				rt_dma_wait(&DmaW_Evt1);
			}
			rt_dma_memcpy((rt_pointerT) Out + ((0)*1250),
				(rt_pointerT) (PULP_Dronet_L1_Memory + 2504) + (1252*(N_Ti % 3)), 1250, RT_DMA_DIR_LOC2EXT, 0, &DmaW_Evt1);
			/* ===================End Write Tile=========================================== */
			N_Ti++;
			/* End Kernel Iteration Loop on Iter space */
		}
		Iter=1;
		In = In + 625;
		Out = Out + 625;
		/* End Kernel Iteration Loop on Input+Output Planes */
	}
	/* =======================Write Last Tile=========================================== */
	rt_dma_wait(&DmaW_Evt1);
	/* ===================End Write Last Tile=========================================== */
}

void MedParConv_3x3_S2_ReLU_5(
		short int * __restrict__ In,
		short int * __restrict__ Filter,
		short int * __restrict__ Out,
		unsigned int Norm,
		short int * __restrict__ Bias,
		Kernel_T *Ker)

{
	/* Local variables used by this kernel */
	rt_dma_copy_t DmaR_Evt1;
	rt_dma_copy_t DmaR_Evt2;
	rt_dma_copy_t DmaR_Evt3;
	rt_dma_copy_t DmaW_Evt1;
	int Iter;
	int Last, NextLast, NextNextLast;
	int N_Ti = 0;
	int Iter1;
	int Last1, NextLast1, NextNextLast1;
	int N_Ti1 = 0;
	int N_TiIp = 0, InPlane, OutPlane=0;
	KerParSetBias_fp_T S_KerArg0, *KerArg0 = &S_KerArg0;
	KerParConv_fp_T S_KerArg1, *KerArg1 = &S_KerArg1;
	KerParReLUMaxPool_fp_T S_KerArg2, *KerArg2 = &S_KerArg2;

	/* Initialize KerArg, Kernel invariant arguments */
	KerArg0->W = (unsigned short int) (13);
	KerArg0->H = (unsigned short int) (13);
	KerArg1->W = (unsigned short int) (25);
	KerArg1->H = (unsigned short int) (25);
	KerArg1->InFeatures = (unsigned short int) (4);
	KerArg1->Norm = (Norm);
	KerArg1->TileIndex = (unsigned short int) (32);
	KerArg1->NTile = (unsigned short int) (0);
	KerArg1->Pad = (v4s) (16843009);
	KerArg2->W = (unsigned short int) (13);
	KerArg2->H = (unsigned short int) (13);
	/* =======================Read First Tile=========================================== */
	/* Initial reads in L2, O_DB or O_BUFF */
	rt_dma_memcpy((rt_pointerT) In+(0), (rt_pointerT) (PULP_Dronet_L1_Memory + 44000)+0, 5000, RT_DMA_DIR_EXT2LOC, 0, &DmaR_Evt1);
	rt_dma_memcpy((rt_pointerT) Filter+(0), (rt_pointerT) (PULP_Dronet_L1_Memory + 0)+0, 13824, RT_DMA_DIR_EXT2LOC, 0, &DmaR_Evt2);
	rt_dma_memcpy((rt_pointerT) Bias+(0), (rt_pointerT) (PULP_Dronet_L1_Memory + 27648)+0, 128, RT_DMA_DIR_EXT2LOC, 0, &DmaR_Evt3);
	/* Wait for BUFF read in L2 */
	rt_dma_wait(&DmaR_Evt3);
	/* ===================End Read First Tile=========================================== */
	/* Kernel Iteration Loop on Iter space */
	for (Iter=0; Iter<3; Iter++) {
		/* Loop Iteration Body on Iter space */
		/* Elaborate Last, Next_Last, Next_Next_Last */
		Last = ((Iter+1) == 3); NextLast = ((Iter+2) == 3); NextNextLast = ((Iter+3) == 3);
		/* =======================Read Next Tile=========================================== */
		rt_dma_wait(&DmaR_Evt2);
		if (!Last) {
			rt_dma_memcpy((rt_pointerT) Filter + ((Iter+1)*13824),
					(rt_pointerT) (PULP_Dronet_L1_Memory + 0) + (13824*((N_Ti+1) % 2)), NextLast?9216:13824, RT_DMA_DIR_EXT2LOC, 0, &DmaR_Evt2);
		}
		/* ===================End Read Next Tile=========================================== */
		/* Call Kernel LOC_Unknown */
		KerArg0->Out = (short int * __restrict__) ((rt_pointerT) (PULP_Dronet_L1_Memory + 27776) + 8112*((N_Ti) % 2));
		KerArg0->OutFeatures = (unsigned short int) (Last?16:24);
		KerArg0->Bias = (short int * __restrict__) ((rt_pointerT) (PULP_Dronet_L1_Memory + 27648) + 0 + (Iter)*48);
		rt_team_fork(gap8_ncore(), (void *) KerParSetBias_fp, (void *) KerArg0);
		/* Kernel Iteration Loop on Iter1 space */
		for (Iter1=0; Iter1<8; Iter1++) {
			/* Loop Iteration Body on Iter1 space */
			/* Elaborate Last, Next_Last, Next_Next_Last */
			Last1 = ((Iter1+1) == 8); NextLast1 = ((Iter1+2) == 8); NextNextLast1 = ((Iter1+3) == 8);
			/* =======================Read Next Tile=========================================== */
			rt_dma_wait(&DmaR_Evt1);
			if (!Last1) {
				rt_dma_memcpy((rt_pointerT) In + ((Iter1+1)*5000),
						(rt_pointerT) (PULP_Dronet_L1_Memory + 44000) + (5000*((N_Ti1+1) % 2)), 5000, RT_DMA_DIR_EXT2LOC, 0, &DmaR_Evt1);
			}
			else if (!Last) {
				rt_dma_memcpy((rt_pointerT) In + (0),
						(rt_pointerT) (PULP_Dronet_L1_Memory + 44000) + (5000*((N_Ti1+1) % 2)), 5000, RT_DMA_DIR_EXT2LOC, 0, &DmaR_Evt1);
			}
			/* ===================End Read Next Tile=========================================== */
			/* Call Kernel LOC_Unknown */
			KerArg1->In = (short int * __restrict__) ((rt_pointerT) (PULP_Dronet_L1_Memory + 44000) + 5000*((N_Ti1) % 2));
			KerArg1->OutFeatures = (unsigned short int) (Last?16:24);
			KerArg1->BaseOutFeature = (unsigned short int) (Iter1)*4;
			KerArg1->Filter = (short int * __restrict__) ((rt_pointerT) (PULP_Dronet_L1_Memory + 0) + 13824*((N_Ti) % 2));
			KerArg1->Out = (short int * __restrict__) ((rt_pointerT) (PULP_Dronet_L1_Memory + 27776) + 8112*((N_Ti) % 2));
			rt_team_fork(gap8_ncore(), (void *) KerParConv3x3Stride2_fp, (void *) KerArg1);
			N_Ti1++;
			/* End Kernel Iteration Loop on Iter1 space */
		}
		Iter1=8;
		/* Call Kernel LOC_Unknown */
		KerArg2->In = (short int * __restrict__) ((rt_pointerT) (PULP_Dronet_L1_Memory + 27776) + 8112*((N_Ti) % 2));
		KerArg2->OutFeatures = (unsigned short int) (Last?16:24);
		KerArg2->Out = (short int * __restrict__) ((rt_pointerT) (PULP_Dronet_L1_Memory + 27776) + 8112*((N_Ti) % 2));
		rt_team_fork(gap8_ncore(), (void *) KerParReLU_fp, (void *) KerArg2);
		/* =======================Write Tile=========================================== */
		if (Iter) {
			rt_dma_wait(&DmaW_Evt1);
		}
		rt_dma_memcpy((rt_pointerT) Out + ((Iter)*8112),
			(rt_pointerT) (PULP_Dronet_L1_Memory + 27776) + (8112*(N_Ti % 2)), Last?5408:8112, RT_DMA_DIR_LOC2EXT, 0, &DmaW_Evt1);
		/* ===================End Write Tile=========================================== */
		N_Ti++;
		/* End Kernel Iteration Loop on Iter space */
	}
	Iter=3;
	/* =======================Write Last Tile=========================================== */
	rt_dma_wait(&DmaW_Evt1);
	/* ===================End Write Last Tile=========================================== */
}

void MedParConv_3x3_S1_6(
		short int * __restrict__ In,
		short int * __restrict__ Filter,
		short int * __restrict__ Out,
		unsigned int Norm,
		short int * __restrict__ Bias,
		Kernel_T *Ker)

{
	/* Local variables used by this kernel */
	rt_dma_copy_t DmaR_Evt1;
	rt_dma_copy_t DmaR_Evt2;
	rt_dma_copy_t DmaR_Evt3;
	rt_dma_copy_t DmaW_Evt1;
	int Iter;
	int Last, NextLast, NextNextLast;
	int N_Ti = 0;
	int Iter1;
	int Last1, NextLast1, NextNextLast1;
	int N_Ti1 = 0;
	int N_TiIp = 0, InPlane, OutPlane=0;
	KerParSetBias_fp_T S_KerArg0, *KerArg0 = &S_KerArg0;
	KerParConv_fp_T S_KerArg1, *KerArg1 = &S_KerArg1;

	/* Initialize KerArg, Kernel invariant arguments */
	KerArg0->W = (unsigned short int) (13);
	KerArg0->H = (unsigned short int) (13);
	KerArg0->OutFeatures = (unsigned short int) (16);
	KerArg1->W = (unsigned short int) (13);
	KerArg1->H = (unsigned short int) (13);
	KerArg1->OutFeatures = (unsigned short int) (16);
	KerArg1->Norm = (Norm);
	KerArg1->TileIndex = (unsigned short int) (64);
	KerArg1->NTile = (unsigned short int) (0);
	KerArg1->Pad = (v4s) (16843009);
	/* =======================Read First Tile=========================================== */
	/* Initial reads in L2, O_DB or O_BUFF */
	rt_dma_memcpy((rt_pointerT) In+(0), (rt_pointerT) (PULP_Dronet_L1_Memory + 47808)+0, 3042, RT_DMA_DIR_EXT2LOC, 0, &DmaR_Evt1);
	rt_dma_memcpy((rt_pointerT) Filter+(0), (rt_pointerT) (PULP_Dronet_L1_Memory + 0)+0, 18432, RT_DMA_DIR_EXT2LOC, 0, &DmaR_Evt2);
	rt_dma_memcpy((rt_pointerT) Bias+(0), (rt_pointerT) (PULP_Dronet_L1_Memory + 36864)+0, 128, RT_DMA_DIR_EXT2LOC, 0, &DmaR_Evt3);
	/* Wait for BUFF read in L2 */
	rt_dma_wait(&DmaR_Evt3);
	/* ===================End Read First Tile=========================================== */
	/* Kernel Iteration Loop on Iter space */
	for (Iter=0; Iter<4; Iter++) {
		/* Loop Iteration Body on Iter space */
		/* Elaborate Last, Next_Last, Next_Next_Last */
		Last = ((Iter+1) == 4); NextLast = ((Iter+2) == 4); NextNextLast = ((Iter+3) == 4);
		/* =======================Read Next Tile=========================================== */
		rt_dma_wait(&DmaR_Evt2);
		if (!Last) {
			rt_dma_memcpy((rt_pointerT) Filter + ((Iter+1)*18432),
					(rt_pointerT) (PULP_Dronet_L1_Memory + 0) + (18432*((N_Ti+1) % 2)), 18432, RT_DMA_DIR_EXT2LOC, 0, &DmaR_Evt2);
		}
		/* ===================End Read Next Tile=========================================== */
		/* Call Kernel LOC_Unknown */
		KerArg0->Out = (short int * __restrict__) ((rt_pointerT) (PULP_Dronet_L1_Memory + 36992) + 5408*((N_Ti) % 2));
		KerArg0->Bias = (short int * __restrict__) ((rt_pointerT) (PULP_Dronet_L1_Memory + 36864) + 0 + (Iter)*32);
		rt_team_fork(gap8_ncore(), (void *) KerParSetBias_fp, (void *) KerArg0);
		/* Kernel Iteration Loop on Iter1 space */
		for (Iter1=0; Iter1<8; Iter1++) {
			/* Loop Iteration Body on Iter1 space */
			/* Elaborate Last, Next_Last, Next_Next_Last */
			Last1 = ((Iter1+1) == 8); NextLast1 = ((Iter1+2) == 8); NextNextLast1 = ((Iter1+3) == 8);
			/* =======================Read Next Tile=========================================== */
			rt_dma_wait(&DmaR_Evt1);
			if (!Last1) {
				rt_dma_memcpy((rt_pointerT) In + ((Iter1+1)*3042),
						(rt_pointerT) (PULP_Dronet_L1_Memory + 47808) + (3044*((N_Ti1+1) % 2)), NextLast1?338:3042, RT_DMA_DIR_EXT2LOC, 0, &DmaR_Evt1);
			}
			else if (!Last) {
				rt_dma_memcpy((rt_pointerT) In + (0),
						(rt_pointerT) (PULP_Dronet_L1_Memory + 47808) + (3044*((N_Ti1+1) % 2)), 3042, RT_DMA_DIR_EXT2LOC, 0, &DmaR_Evt1);
			}
			/* ===================End Read Next Tile=========================================== */
			/* Call Kernel LOC_Unknown */
			KerArg1->In = (short int * __restrict__) ((rt_pointerT) (PULP_Dronet_L1_Memory + 47808) + 3044*((N_Ti1) % 2));
			KerArg1->InFeatures = (unsigned short int) (Last1?1:9);
			KerArg1->BaseOutFeature = (unsigned short int) (Iter1)*9;
			KerArg1->Filter = (short int * __restrict__) ((rt_pointerT) (PULP_Dronet_L1_Memory + 0) + 18432*((N_Ti) % 2));
			KerArg1->Out = (short int * __restrict__) ((rt_pointerT) (PULP_Dronet_L1_Memory + 36992) + 5408*((N_Ti) % 2));
			rt_team_fork(gap8_ncore(), (void *) KerParConv3x3Stride1_fp, (void *) KerArg1);
			N_Ti1++;
			/* End Kernel Iteration Loop on Iter1 space */
		}
		Iter1=8;
		/* =======================Write Tile=========================================== */
		if (Iter) {
			rt_dma_wait(&DmaW_Evt1);
		}
		rt_dma_memcpy((rt_pointerT) Out + ((Iter)*5408),
			(rt_pointerT) (PULP_Dronet_L1_Memory + 36992) + (5408*(N_Ti % 2)), 5408, RT_DMA_DIR_LOC2EXT, 0, &DmaW_Evt1);
		/* ===================End Write Tile=========================================== */
		N_Ti++;
		/* End Kernel Iteration Loop on Iter space */
	}
	Iter=4;
	/* =======================Write Last Tile=========================================== */
	rt_dma_wait(&DmaW_Evt1);
	/* ===================End Write Last Tile=========================================== */
}

void MedParConv_1x1_S2_7(
		short int * __restrict__ In,
		short int * __restrict__ Filter,
		short int * __restrict__ Out,
		unsigned int Norm,
		short int * __restrict__ Bias,
		Kernel_T *Ker)

{
	/* Local variables used by this kernel */
	rt_dma_copy_t DmaR_Evt1;
	rt_dma_copy_t DmaR_Evt2;
	rt_dma_copy_t DmaR_Evt3;
	rt_dma_copy_t DmaW_Evt1;
	int Iter;
	int Last, NextLast, NextNextLast;
	int N_Ti = 0;
	int Iter1;
	int Last1, NextLast1, NextNextLast1;
	int N_Ti1 = 0;
	int N_TiIp = 0, InPlane, OutPlane=0;
	KerParSetBias_fp_T S_KerArg0, *KerArg0 = &S_KerArg0;
	KerParConv_fp_T S_KerArg1, *KerArg1 = &S_KerArg1;

	/* Initialize KerArg, Kernel invariant arguments */
	KerArg0->W = (unsigned short int) (13);
	KerArg0->H = (unsigned short int) (13);
	KerArg0->OutFeatures = (unsigned short int) (64);
	KerArg1->W = (unsigned short int) (25);
	KerArg1->H = (unsigned short int) (25);
	KerArg1->OutFeatures = (unsigned short int) (64);
	KerArg1->Norm = (Norm);
	KerArg1->TileIndex = (unsigned short int) (32);
	KerArg1->NTile = (unsigned short int) (0);
	KerArg1->Pad = (v4s) (0);
	/* =======================Read First Tile=========================================== */
	/* Initial reads in L2, O_DB or O_BUFF */
	rt_dma_memcpy((rt_pointerT) In+(0), (rt_pointerT) (PULP_Dronet_L1_Memory + 25856)+0, 13750, RT_DMA_DIR_EXT2LOC, 0, &DmaR_Evt1);
	rt_dma_memcpy((rt_pointerT) Filter+(0), (rt_pointerT) (PULP_Dronet_L1_Memory + 0)+0, 4096, RT_DMA_DIR_EXT2LOC, 0, &DmaR_Evt2);
	rt_dma_memcpy((rt_pointerT) Bias+(0), (rt_pointerT) (PULP_Dronet_L1_Memory + 4096)+0, 128, RT_DMA_DIR_EXT2LOC, 0, &DmaR_Evt3);
	/* Wait for BUFF read in L2 */
	rt_dma_wait(&DmaR_Evt2);
	rt_dma_wait(&DmaR_Evt3);
	/* ===================End Read First Tile=========================================== */
	/* Kernel Iteration Loop on Iter space */
	Iter=0; {
		/* Loop Iteration Body on Iter space */
		/* Elaborate Last, Next_Last, Next_Next_Last */
		Last = ((Iter+1) == 1); NextLast = ((Iter+2) == 1); NextNextLast = ((Iter+3) == 1);
		/* Call Kernel LOC_Unknown */
		KerArg0->Out = (short int * __restrict__) ((rt_pointerT) (PULP_Dronet_L1_Memory + 4224) + 0 + (0)*21632);
		KerArg0->Bias = (short int * __restrict__) ((rt_pointerT) (PULP_Dronet_L1_Memory + 4096) + 0 + (0)*128);
		rt_team_fork(gap8_ncore(), (void *) KerParSetBias_fp, (void *) KerArg0);
		/* Kernel Iteration Loop on Iter1 space */
		for (Iter1=0; Iter1<3; Iter1++) {
			/* Loop Iteration Body on Iter1 space */
			/* Elaborate Last, Next_Last, Next_Next_Last */
			Last1 = ((Iter1+1) == 3); NextLast1 = ((Iter1+2) == 3); NextNextLast1 = ((Iter1+3) == 3);
			/* =======================Read Next Tile=========================================== */
			rt_dma_wait(&DmaR_Evt1);
			if (!Last1) {
				rt_dma_memcpy((rt_pointerT) In + ((Iter1+1)*13750),
						(rt_pointerT) (PULP_Dronet_L1_Memory + 25856) + (13752*((N_Ti1+1) % 2)), NextLast1?12500:13750, RT_DMA_DIR_EXT2LOC, 0, &DmaR_Evt1);
			}
			else if (!Last) {
				rt_dma_memcpy((rt_pointerT) In + (0),
						(rt_pointerT) (PULP_Dronet_L1_Memory + 25856) + (13752*((N_Ti1+1) % 2)), 13750, RT_DMA_DIR_EXT2LOC, 0, &DmaR_Evt1);
			}
			/* ===================End Read Next Tile=========================================== */
			/* Call Kernel LOC_Unknown */
			KerArg1->In = (short int * __restrict__) ((rt_pointerT) (PULP_Dronet_L1_Memory + 25856) + 13752*((N_Ti1) % 2));
			KerArg1->InFeatures = (unsigned short int) (Last1?10:11);
			KerArg1->BaseOutFeature = (unsigned short int) (Iter1)*11;
			KerArg1->Filter = (short int * __restrict__) ((rt_pointerT) (PULP_Dronet_L1_Memory + 0) + 0 + (0)*4096);
			KerArg1->Out = (short int * __restrict__) ((rt_pointerT) (PULP_Dronet_L1_Memory + 4224) + 0 + (0)*21632);
			rt_team_fork(gap8_ncore(), (void *) KerParConv1x1Stride2_fp, (void *) KerArg1);
			N_Ti1++;
			/* End Kernel Iteration Loop on Iter1 space */
		}
		Iter1=3;
		N_Ti++;
		/* End Kernel Iteration Loop on Iter space */
	}
	Iter=1;
	/* =======================Write Last Tile=========================================== */
	rt_dma_memcpy((rt_pointerT) Out + (0),
		(rt_pointerT) (PULP_Dronet_L1_Memory + 4224) + 0, 21632, RT_DMA_DIR_LOC2EXT, 0, &DmaW_Evt1);
	rt_dma_wait(&DmaW_Evt1);
	/* ===================End Write Last Tile=========================================== */
}

void AddFeatureMaps_SW_2(
		short int * __restrict__ In,
		short int * __restrict__ Out,
		Kernel_T *Ker)

{
	/* Local variables used by this kernel */
	rt_dma_copy_t DmaR_Evt1;
	rt_dma_copy_t DmaR_Evt2;
	rt_dma_copy_t DmaW_Evt1;
	int Iter;
	int Last, NextLast, NextNextLast;
	int N_Ti = 0;
	int N_TiIp = 0, InPlane, OutPlane=0;
	KerAddFM_fpT S_KerArg0, *KerArg0 = &S_KerArg0;

	/* Initialize KerArg, Kernel invariant arguments */
	KerArg0->W = (int) (13);
	KerArg0->H = (int) (13);
	/* =======================Read First Tile=========================================== */
	/* Initial reads in L2, O_DB or O_BUFF */
	rt_dma_memcpy((rt_pointerT) In+(0), (rt_pointerT) (PULP_Dronet_L1_Memory + 0)+0, 21632, RT_DMA_DIR_EXT2LOC, 0, &DmaR_Evt1);
	rt_dma_memcpy((rt_pointerT) Out+(0), (rt_pointerT) (PULP_Dronet_L1_Memory + 21632)+0, 21632, RT_DMA_DIR_EXT2LOC, 0, &DmaR_Evt2);
	/* Wait for BUFF read in L2 */
	rt_dma_wait(&DmaR_Evt1);
	rt_dma_wait(&DmaR_Evt2);
	/* ===================End Read First Tile=========================================== */
	/* Kernel Iteration Loop on Input+Output Planes */
	for (OutPlane=0; OutPlane<64; OutPlane++) {
		int LastOutPlane = ((OutPlane+1) == 64), NextLastOutPlane = ((OutPlane+2) == 64);
		/* Kernel Iteration Loop on Iter space */
		Iter=0; {
			/* Loop Iteration Body on Iter space */
			/* Elaborate Last, Next_Last, Next_Next_Last */
			Last = ((Iter+1) == 1); NextLast = ((Iter+2) == 1); NextNextLast = ((Iter+3) == 1);
			/* Call Kernel LOC_INNER_LOOP */
			KerArg0->In = (Word16 * __restrict__) ((rt_pointerT) (PULP_Dronet_L1_Memory + 0) + OutPlane*338 + (0)*338);
			KerArg0->Out = (Word16 * __restrict__) ((rt_pointerT) (PULP_Dronet_L1_Memory + 21632) + OutPlane*338 + (0)*338);
			rt_team_fork(gap8_ncore(), (void *) KerAddFM_fp, (void *) KerArg0);
			N_Ti++;
			/* End Kernel Iteration Loop on Iter space */
		}
		Iter=1;
		/* End Kernel Iteration Loop on Input+Output Planes */
	}
	/* =======================Write Last Tile=========================================== */
	rt_dma_memcpy((rt_pointerT) Out + (0),
		(rt_pointerT) (PULP_Dronet_L1_Memory + 21632) + 0, 21632, RT_DMA_DIR_LOC2EXT, 0, &DmaW_Evt1);
	rt_dma_wait(&DmaW_Evt1);
	/* ===================End Write Last Tile=========================================== */
}

void ReLU_SW_3(
		short int * __restrict__ In,
		short int * __restrict__ Out,
		Kernel_T *Ker)

{
	/* Local variables used by this kernel */
	rt_dma_copy_t DmaR_Evt1;
	rt_dma_copy_t DmaR_Evt2;
	rt_dma_copy_t DmaW_Evt1;
	int Iter;
	int Last, NextLast, NextNextLast;
	int N_Ti = 0;
	int N_TiIp = 0, InPlane, OutPlane=0;
	KerReLUMaxPool2x2_fpT S_KerArg0, *KerArg0 = &S_KerArg0;

	/* Initialize KerArg, Kernel invariant arguments */
	KerArg0->W = (int) (13);
	KerArg0->H = (int) (13);
	/* =======================Read First Tile=========================================== */
	/* Initial reads in L2, O_DB or O_BUFF */
	rt_dma_memcpy((rt_pointerT) In+(0), (rt_pointerT) (PULP_Dronet_L1_Memory + 0)+0, 21632, RT_DMA_DIR_EXT2LOC, 0, &DmaR_Evt1);
	rt_dma_memcpy((rt_pointerT) Out+(0), (rt_pointerT) (PULP_Dronet_L1_Memory + 21632)+0, 21632, RT_DMA_DIR_EXT2LOC, 0, &DmaR_Evt2);
	/* Wait for BUFF read in L2 */
	rt_dma_wait(&DmaR_Evt1);
	rt_dma_wait(&DmaR_Evt2);
	/* ===================End Read First Tile=========================================== */
	/* Kernel Iteration Loop on Input+Output Planes */
	for (OutPlane=0; OutPlane<64; OutPlane++) {
		int LastOutPlane = ((OutPlane+1) == 64), NextLastOutPlane = ((OutPlane+2) == 64);
		/* Kernel Iteration Loop on Iter space */
		Iter=0; {
			/* Loop Iteration Body on Iter space */
			/* Elaborate Last, Next_Last, Next_Next_Last */
			Last = ((Iter+1) == 1); NextLast = ((Iter+2) == 1); NextNextLast = ((Iter+3) == 1);
			/* Call Kernel LOC_INNER_LOOP */
			KerArg0->In = (short int * __restrict__) ((rt_pointerT) (PULP_Dronet_L1_Memory + 0) + OutPlane*338 + (0)*338);
			KerArg0->Out = (short int * __restrict__) ((rt_pointerT) (PULP_Dronet_L1_Memory + 21632) + OutPlane*338 + (0)*338);
			rt_team_fork(gap8_ncore(), (void *) KerReLU_fp, (void *) KerArg0);
			N_Ti++;
			/* End Kernel Iteration Loop on Iter space */
		}
		Iter=1;
		/* End Kernel Iteration Loop on Input+Output Planes */
	}
	/* =======================Write Last Tile=========================================== */
	rt_dma_memcpy((rt_pointerT) Out + (0),
		(rt_pointerT) (PULP_Dronet_L1_Memory + 21632) + 0, 21632, RT_DMA_DIR_LOC2EXT, 0, &DmaW_Evt1);
	rt_dma_wait(&DmaW_Evt1);
	/* ===================End Write Last Tile=========================================== */
}

void MedParConv_3x3_S2_ReLU_8(
		short int * __restrict__ In,
		short int * __restrict__ Filter,
		short int * __restrict__ Out,
		unsigned int Norm,
		short int * __restrict__ Bias,
		Kernel_T *Ker)

{
	/* Local variables used by this kernel */
	rt_dma_copy_t DmaR_Evt1;
	rt_dma_copy_t DmaR_Evt2;
	rt_dma_copy_t DmaR_Evt3;
	rt_dma_copy_t DmaW_Evt1;
	int Iter;
	int Last, NextLast, NextNextLast;
	int N_Ti = 0;
	int Iter1;
	int Last1, NextLast1, NextNextLast1;
	int N_Ti1 = 0;
	int N_TiIp = 0, InPlane, OutPlane=0;
	KerParSetBias_fp_T S_KerArg0, *KerArg0 = &S_KerArg0;
	KerParConv_fp_T S_KerArg1, *KerArg1 = &S_KerArg1;
	KerParReLUMaxPool_fp_T S_KerArg2, *KerArg2 = &S_KerArg2;

	/* Initialize KerArg, Kernel invariant arguments */
	KerArg0->W = (unsigned short int) (7);
	KerArg0->H = (unsigned short int) (7);
	KerArg0->OutFeatures = (unsigned short int) (16);
	KerArg1->W = (unsigned short int) (13);
	KerArg1->H = (unsigned short int) (13);
	KerArg1->OutFeatures = (unsigned short int) (16);
	KerArg1->Norm = (Norm);
	KerArg1->TileIndex = (unsigned short int) (64);
	KerArg1->NTile = (unsigned short int) (0);
	KerArg1->Pad = (v4s) (16843009);
	KerArg2->W = (unsigned short int) (7);
	KerArg2->H = (unsigned short int) (7);
	KerArg2->OutFeatures = (unsigned short int) (16);
	/* =======================Read First Tile=========================================== */
	/* Initial reads in L2, O_DB or O_BUFF */
	rt_dma_memcpy((rt_pointerT) In+(0), (rt_pointerT) (PULP_Dronet_L1_Memory + 40256)+0, 6760, RT_DMA_DIR_EXT2LOC, 0, &DmaR_Evt1);
	rt_dma_memcpy((rt_pointerT) Filter+(0), (rt_pointerT) (PULP_Dronet_L1_Memory + 0)+0, 18432, RT_DMA_DIR_EXT2LOC, 0, &DmaR_Evt2);
	rt_dma_memcpy((rt_pointerT) Bias+(0), (rt_pointerT) (PULP_Dronet_L1_Memory + 36864)+0, 256, RT_DMA_DIR_EXT2LOC, 0, &DmaR_Evt3);
	/* Wait for BUFF read in L2 */
	rt_dma_wait(&DmaR_Evt3);
	/* ===================End Read First Tile=========================================== */
	/* Kernel Iteration Loop on Iter space */
	for (Iter=0; Iter<8; Iter++) {
		/* Loop Iteration Body on Iter space */
		/* Elaborate Last, Next_Last, Next_Next_Last */
		Last = ((Iter+1) == 8); NextLast = ((Iter+2) == 8); NextNextLast = ((Iter+3) == 8);
		/* =======================Read Next Tile=========================================== */
		rt_dma_wait(&DmaR_Evt2);
		if (!Last) {
			rt_dma_memcpy((rt_pointerT) Filter + ((Iter+1)*18432),
					(rt_pointerT) (PULP_Dronet_L1_Memory + 0) + (18432*((N_Ti+1) % 2)), 18432, RT_DMA_DIR_EXT2LOC, 0, &DmaR_Evt2);
		}
		/* ===================End Read Next Tile=========================================== */
		/* Call Kernel LOC_Unknown */
		KerArg0->Out = (short int * __restrict__) ((rt_pointerT) (PULP_Dronet_L1_Memory + 37120) + 1568*((N_Ti) % 2));
		KerArg0->Bias = (short int * __restrict__) ((rt_pointerT) (PULP_Dronet_L1_Memory + 36864) + 0 + (Iter)*32);
		rt_team_fork(gap8_ncore(), (void *) KerParSetBias_fp, (void *) KerArg0);
		/* Kernel Iteration Loop on Iter1 space */
		for (Iter1=0; Iter1<4; Iter1++) {
			/* Loop Iteration Body on Iter1 space */
			/* Elaborate Last, Next_Last, Next_Next_Last */
			Last1 = ((Iter1+1) == 4); NextLast1 = ((Iter1+2) == 4); NextNextLast1 = ((Iter1+3) == 4);
			/* =======================Read Next Tile=========================================== */
			rt_dma_wait(&DmaR_Evt1);
			if (!Last1) {
				rt_dma_memcpy((rt_pointerT) In + ((Iter1+1)*6760),
						(rt_pointerT) (PULP_Dronet_L1_Memory + 40256) + (6760*((N_Ti1+1) % 2)), NextLast1?1352:6760, RT_DMA_DIR_EXT2LOC, 0, &DmaR_Evt1);
			}
			else if (!Last) {
				rt_dma_memcpy((rt_pointerT) In + (0),
						(rt_pointerT) (PULP_Dronet_L1_Memory + 40256) + (6760*((N_Ti1+1) % 2)), 6760, RT_DMA_DIR_EXT2LOC, 0, &DmaR_Evt1);
			}
			/* ===================End Read Next Tile=========================================== */
			/* Call Kernel LOC_Unknown */
			KerArg1->In = (short int * __restrict__) ((rt_pointerT) (PULP_Dronet_L1_Memory + 40256) + 6760*((N_Ti1) % 2));
			KerArg1->InFeatures = (unsigned short int) (Last1?4:20);
			KerArg1->BaseOutFeature = (unsigned short int) (Iter1)*20;
			KerArg1->Filter = (short int * __restrict__) ((rt_pointerT) (PULP_Dronet_L1_Memory + 0) + 18432*((N_Ti) % 2));
			KerArg1->Out = (short int * __restrict__) ((rt_pointerT) (PULP_Dronet_L1_Memory + 37120) + 1568*((N_Ti) % 2));
			rt_team_fork(gap8_ncore(), (void *) KerParConv3x3Stride2_fp, (void *) KerArg1);
			N_Ti1++;
			/* End Kernel Iteration Loop on Iter1 space */
		}
		Iter1=4;
		/* Call Kernel LOC_Unknown */
		KerArg2->In = (short int * __restrict__) ((rt_pointerT) (PULP_Dronet_L1_Memory + 37120) + 1568*((N_Ti) % 2));
		KerArg2->Out = (short int * __restrict__) ((rt_pointerT) (PULP_Dronet_L1_Memory + 37120) + 1568*((N_Ti) % 2));
		rt_team_fork(gap8_ncore(), (void *) KerParReLU_fp, (void *) KerArg2);
		/* =======================Write Tile=========================================== */
		if (Iter) {
			rt_dma_wait(&DmaW_Evt1);
		}
		rt_dma_memcpy((rt_pointerT) Out + ((Iter)*1568),
			(rt_pointerT) (PULP_Dronet_L1_Memory + 37120) + (1568*(N_Ti % 2)), 1568, RT_DMA_DIR_LOC2EXT, 0, &DmaW_Evt1);
		/* ===================End Write Tile=========================================== */
		N_Ti++;
		/* End Kernel Iteration Loop on Iter space */
	}
	Iter=8;
	/* =======================Write Last Tile=========================================== */
	rt_dma_wait(&DmaW_Evt1);
	/* ===================End Write Last Tile=========================================== */
}

void MedParConv_3x3_S1_9(
		short int * __restrict__ In,
		short int * __restrict__ Filter,
		short int * __restrict__ Out,
		unsigned int Norm,
		short int * __restrict__ Bias,
		Kernel_T *Ker)

{
	/* Local variables used by this kernel */
	rt_dma_copy_t DmaR_Evt1;
	rt_dma_copy_t DmaR_Evt2;
	rt_dma_copy_t DmaR_Evt3;
	rt_dma_copy_t DmaW_Evt1;
	int Iter;
	int Last, NextLast, NextNextLast;
	int N_Ti = 0;
	int Iter1;
	int Last1, NextLast1, NextNextLast1;
	int N_Ti1 = 0;
	int N_TiIp = 0, InPlane, OutPlane=0;
	KerParSetBias_fp_T S_KerArg0, *KerArg0 = &S_KerArg0;
	KerParConv_fp_T S_KerArg1, *KerArg1 = &S_KerArg1;

	/* Initialize KerArg, Kernel invariant arguments */
	KerArg0->W = (unsigned short int) (7);
	KerArg0->H = (unsigned short int) (7);
	KerArg0->OutFeatures = (unsigned short int) (8);
	KerArg1->W = (unsigned short int) (7);
	KerArg1->H = (unsigned short int) (7);
	KerArg1->InFeatures = (unsigned short int) (128);
	KerArg1->OutFeatures = (unsigned short int) (8);
	KerArg1->Norm = (Norm);
	KerArg1->TileIndex = (unsigned short int) (128);
	KerArg1->NTile = (unsigned short int) (0);
	KerArg1->Pad = (v4s) (16843009);
	/* =======================Read First Tile=========================================== */
	/* Initial reads in L2, O_DB or O_BUFF */
	rt_dma_memcpy((rt_pointerT) In+(0), (rt_pointerT) (PULP_Dronet_L1_Memory + 38688)+0, 12544, RT_DMA_DIR_EXT2LOC, 0, &DmaR_Evt1);
	rt_dma_memcpy((rt_pointerT) Filter+(0), (rt_pointerT) (PULP_Dronet_L1_Memory + 0)+0, 18432, RT_DMA_DIR_EXT2LOC, 0, &DmaR_Evt2);
	rt_dma_memcpy((rt_pointerT) Bias+(0), (rt_pointerT) (PULP_Dronet_L1_Memory + 36864)+0, 256, RT_DMA_DIR_EXT2LOC, 0, &DmaR_Evt3);
	/* Wait for BUFF read in L2 */
	rt_dma_wait(&DmaR_Evt1);
	rt_dma_wait(&DmaR_Evt3);
	/* ===================End Read First Tile=========================================== */
	/* Kernel Iteration Loop on Iter space */
	for (Iter=0; Iter<16; Iter++) {
		/* Loop Iteration Body on Iter space */
		/* Elaborate Last, Next_Last, Next_Next_Last */
		Last = ((Iter+1) == 16); NextLast = ((Iter+2) == 16); NextNextLast = ((Iter+3) == 16);
		/* =======================Read Next Tile=========================================== */
		rt_dma_wait(&DmaR_Evt2);
		if (!Last) {
			rt_dma_memcpy((rt_pointerT) Filter + ((Iter+1)*18432),
					(rt_pointerT) (PULP_Dronet_L1_Memory + 0) + (18432*((N_Ti+1) % 2)), 18432, RT_DMA_DIR_EXT2LOC, 0, &DmaR_Evt2);
		}
		/* ===================End Read Next Tile=========================================== */
		/* Call Kernel LOC_Unknown */
		KerArg0->Out = (short int * __restrict__) ((rt_pointerT) (PULP_Dronet_L1_Memory + 37120) + 784*((N_Ti) % 2));
		KerArg0->Bias = (short int * __restrict__) ((rt_pointerT) (PULP_Dronet_L1_Memory + 36864) + 0 + (Iter)*16);
		rt_team_fork(gap8_ncore(), (void *) KerParSetBias_fp, (void *) KerArg0);
		/* Kernel Iteration Loop on Iter1 space */
		Iter1=0; {
			/* Loop Iteration Body on Iter1 space */
			/* Elaborate Last, Next_Last, Next_Next_Last */
			Last1 = ((Iter1+1) == 1); NextLast1 = ((Iter1+2) == 1); NextNextLast1 = ((Iter1+3) == 1);
			/* Call Kernel LOC_Unknown */
			KerArg1->In = (short int * __restrict__) ((rt_pointerT) (PULP_Dronet_L1_Memory + 38688) + 0 + (0)*12544);
			KerArg1->BaseOutFeature = (unsigned short int) (0)*128;
			KerArg1->Filter = (short int * __restrict__) ((rt_pointerT) (PULP_Dronet_L1_Memory + 0) + 18432*((N_Ti) % 2));
			KerArg1->Out = (short int * __restrict__) ((rt_pointerT) (PULP_Dronet_L1_Memory + 37120) + 784*((N_Ti) % 2));
			rt_team_fork(gap8_ncore(), (void *) KerParConv3x3Stride1_fp, (void *) KerArg1);
			N_Ti1++;
			/* End Kernel Iteration Loop on Iter1 space */
		}
		Iter1=1;
		/* =======================Write Tile=========================================== */
		if (Iter) {
			rt_dma_wait(&DmaW_Evt1);
		}
		rt_dma_memcpy((rt_pointerT) Out + ((Iter)*784),
			(rt_pointerT) (PULP_Dronet_L1_Memory + 37120) + (784*(N_Ti % 2)), 784, RT_DMA_DIR_LOC2EXT, 0, &DmaW_Evt1);
		/* ===================End Write Tile=========================================== */
		N_Ti++;
		/* End Kernel Iteration Loop on Iter space */
	}
	Iter=16;
	/* =======================Write Last Tile=========================================== */
	rt_dma_wait(&DmaW_Evt1);
	/* ===================End Write Last Tile=========================================== */
}

void MedParConv_1x1_S1_ReLU_10(
		short int * __restrict__ In,
		short int * __restrict__ Filter,
		short int * __restrict__ Out,
		unsigned int Norm,
		short int * __restrict__ Bias,
		Kernel_T *Ker)

{
	/* Local variables used by this kernel */
	rt_dma_copy_t DmaR_Evt1;
	rt_dma_copy_t DmaR_Evt2;
	rt_dma_copy_t DmaR_Evt3;
	rt_dma_copy_t DmaW_Evt1;
	int Iter;
	int Last, NextLast, NextNextLast;
	int N_Ti = 0;
	int Iter1;
	int Last1, NextLast1, NextNextLast1;
	int N_Ti1 = 0;
	int N_TiIp = 0, InPlane, OutPlane=0;
	KerParSetBias_fp_T S_KerArg0, *KerArg0 = &S_KerArg0;
	KerParConv_fp_T S_KerArg1, *KerArg1 = &S_KerArg1;

	/* Initialize KerArg, Kernel invariant arguments */
	KerArg0->W = (unsigned short int) (7);
	KerArg0->H = (unsigned short int) (7);
	KerArg0->OutFeatures = (unsigned short int) (128);
	KerArg1->W = (unsigned short int) (13);
	KerArg1->H = (unsigned short int) (13);
	KerArg1->InFeatures = (unsigned short int) (64);
	KerArg1->OutFeatures = (unsigned short int) (128);
	KerArg1->Norm = (Norm);
	KerArg1->TileIndex = (unsigned short int) (64);
	KerArg1->NTile = (unsigned short int) (0);
	KerArg1->Pad = (v4s) (0);
	/* =======================Read First Tile=========================================== */
	/* Initial reads in L2, O_DB or O_BUFF */
	rt_dma_memcpy((rt_pointerT) In+(0), (rt_pointerT) (PULP_Dronet_L1_Memory + 29184)+0, 21632, RT_DMA_DIR_EXT2LOC, 0, &DmaR_Evt1);
	rt_dma_memcpy((rt_pointerT) Filter+(0), (rt_pointerT) (PULP_Dronet_L1_Memory + 0)+0, 16384, RT_DMA_DIR_EXT2LOC, 0, &DmaR_Evt2);
	rt_dma_memcpy((rt_pointerT) Bias+(0), (rt_pointerT) (PULP_Dronet_L1_Memory + 16384)+0, 256, RT_DMA_DIR_EXT2LOC, 0, &DmaR_Evt3);
	/* Wait for BUFF read in L2 */
	rt_dma_wait(&DmaR_Evt1);
	rt_dma_wait(&DmaR_Evt2);
	rt_dma_wait(&DmaR_Evt3);
	/* ===================End Read First Tile=========================================== */
	/* Kernel Iteration Loop on Iter space */
	Iter=0; {
		/* Loop Iteration Body on Iter space */
		/* Elaborate Last, Next_Last, Next_Next_Last */
		Last = ((Iter+1) == 1); NextLast = ((Iter+2) == 1); NextNextLast = ((Iter+3) == 1);
		/* Call Kernel LOC_Unknown */
		KerArg0->Out = (short int * __restrict__) ((rt_pointerT) (PULP_Dronet_L1_Memory + 16640) + 0 + (0)*12544);
		KerArg0->Bias = (short int * __restrict__) ((rt_pointerT) (PULP_Dronet_L1_Memory + 16384) + 0 + (0)*256);
		rt_team_fork(gap8_ncore(), (void *) KerParSetBias_fp, (void *) KerArg0);
		/* Kernel Iteration Loop on Iter1 space */
		Iter1=0; {
			/* Loop Iteration Body on Iter1 space */
			/* Elaborate Last, Next_Last, Next_Next_Last */
			Last1 = ((Iter1+1) == 1); NextLast1 = ((Iter1+2) == 1); NextNextLast1 = ((Iter1+3) == 1);
			/* Call Kernel LOC_Unknown */
			KerArg1->In = (short int * __restrict__) ((rt_pointerT) (PULP_Dronet_L1_Memory + 29184) + 0 + (0)*21632);
			KerArg1->BaseOutFeature = (unsigned short int) (0)*64;
			KerArg1->Filter = (short int * __restrict__) ((rt_pointerT) (PULP_Dronet_L1_Memory + 0) + 0 + (0)*16384);
			KerArg1->Out = (short int * __restrict__) ((rt_pointerT) (PULP_Dronet_L1_Memory + 16640) + 0 + (0)*12544);
			rt_team_fork(gap8_ncore(), (void *) KerParConv1x1Stride2_fp, (void *) KerArg1);
			N_Ti1++;
			/* End Kernel Iteration Loop on Iter1 space */
		}
		Iter1=1;
		N_Ti++;
		/* End Kernel Iteration Loop on Iter space */
	}
	Iter=1;
	/* =======================Write Last Tile=========================================== */
	rt_dma_memcpy((rt_pointerT) Out + (0),
		(rt_pointerT) (PULP_Dronet_L1_Memory + 16640) + 0, 12544, RT_DMA_DIR_LOC2EXT, 0, &DmaW_Evt1);
	rt_dma_wait(&DmaW_Evt1);
	/* ===================End Write Last Tile=========================================== */
}

void AddFeatureMapsReLu_SW_3(
		short int * __restrict__ In,
		short int * __restrict__ Out,
		Kernel_T *Ker)

{
	/* Local variables used by this kernel */
	rt_dma_copy_t DmaR_Evt1;
	rt_dma_copy_t DmaR_Evt2;
	rt_dma_copy_t DmaW_Evt1;
	int Iter;
	int Last, NextLast, NextNextLast;
	int N_Ti = 0;
	int N_TiIp = 0, InPlane, OutPlane=0;
	KerAddFM_fpT S_KerArg0, *KerArg0 = &S_KerArg0;

	/* Initialize KerArg, Kernel invariant arguments */
	KerArg0->W = (int) (7);
	KerArg0->H = (int) (7);
	/* =======================Read First Tile=========================================== */
	/* Initial reads in L2, O_DB or O_BUFF */
	rt_dma_memcpy((rt_pointerT) In+(0), (rt_pointerT) (PULP_Dronet_L1_Memory + 0)+0, 12544, RT_DMA_DIR_EXT2LOC, 0, &DmaR_Evt1);
	rt_dma_memcpy((rt_pointerT) Out+(0), (rt_pointerT) (PULP_Dronet_L1_Memory + 12544)+0, 12544, RT_DMA_DIR_EXT2LOC, 0, &DmaR_Evt2);
	/* Wait for BUFF read in L2 */
	rt_dma_wait(&DmaR_Evt1);
	rt_dma_wait(&DmaR_Evt2);
	/* ===================End Read First Tile=========================================== */
	/* Kernel Iteration Loop on Input+Output Planes */
	for (OutPlane=0; OutPlane<128; OutPlane++) {
		int LastOutPlane = ((OutPlane+1) == 128), NextLastOutPlane = ((OutPlane+2) == 128);
		/* Kernel Iteration Loop on Iter space */
		Iter=0; {
			/* Loop Iteration Body on Iter space */
			/* Elaborate Last, Next_Last, Next_Next_Last */
			Last = ((Iter+1) == 1); NextLast = ((Iter+2) == 1); NextNextLast = ((Iter+3) == 1);
			/* Call Kernel LOC_INNER_LOOP */
			KerArg0->In = (Word16 * __restrict__) ((rt_pointerT) (PULP_Dronet_L1_Memory + 0) + OutPlane*98 + (0)*98);
			KerArg0->Out = (Word16 * __restrict__) ((rt_pointerT) (PULP_Dronet_L1_Memory + 12544) + OutPlane*98 + (0)*98);
			rt_team_fork(gap8_ncore(), (void *) KerAddFMReLu_fp, (void *) KerArg0);
			N_Ti++;
			/* End Kernel Iteration Loop on Iter space */
		}
		Iter=1;
		/* End Kernel Iteration Loop on Input+Output Planes */
	}
	/* =======================Write Last Tile=========================================== */
	rt_dma_memcpy((rt_pointerT) Out + (0),
		(rt_pointerT) (PULP_Dronet_L1_Memory + 12544) + 0, 12544, RT_DMA_DIR_LOC2EXT, 0, &DmaW_Evt1);
	rt_dma_wait(&DmaW_Evt1);
	/* ===================End Write Last Tile=========================================== */
}

void LinearLayer_SW_1(
		Word16 * __restrict__ In,
		Word16 * __restrict__ Filter,
		unsigned int NormFilter,
		Word16 * __restrict__ Bias,
		unsigned int NormBias,
		Word16 * __restrict__ Out,
		int OutSize,
		Kernel_T *Ker)

{
	/* Local variables used by this kernel */
	rt_dma_copy_t DmaR_Evt1;
	rt_dma_copy_t DmaR_Evt2;
	rt_dma_copy_t DmaR_Evt3;
	rt_dma_copy_t DmaW_Evt1;
	int Iter;
	int Last, NextLast, NextNextLast;
	int N_Ti = 0;
	int N_TiIp = 0, InPlane, OutPlane=0;
	KerLinearLayer_fpT S_KerArg0, *KerArg0 = &S_KerArg0;

	/* Initialize KerArg, Kernel invariant arguments */
	KerArg0->InSize = (int) (6272);
	KerArg0->NormFilter = (NormFilter);
	KerArg0->NormBias = (NormBias);
	KerArg0->OutSize = (int) (1);
	/* =======================Read First Tile=========================================== */
	/* Initial reads in L2, O_DB or O_BUFF */
	rt_dma_memcpy((rt_pointerT) In+0, (rt_pointerT) (PULP_Dronet_L1_Memory + 0)+0, 12544, RT_DMA_DIR_EXT2LOC, 0, &DmaR_Evt1);
	rt_dma_memcpy((rt_pointerT) Bias+(0), (rt_pointerT) (PULP_Dronet_L1_Memory + 12544)+0, 2, RT_DMA_DIR_EXT2LOC, 0, &DmaR_Evt2);
	rt_dma_memcpy((rt_pointerT) Filter+(0), (rt_pointerT) (PULP_Dronet_L1_Memory + 12548)+0, 12544, RT_DMA_DIR_EXT2LOC, 0, &DmaR_Evt3);
	/* Wait for BUFF read in L2 */
	rt_dma_wait(&DmaR_Evt1);
	rt_dma_wait(&DmaR_Evt2);
	rt_dma_wait(&DmaR_Evt3);
	/* ===================End Read First Tile=========================================== */
	/* Kernel Iteration Loop on Iter space */
	Iter=0; {
		/* Loop Iteration Body on Iter space */
		/* Elaborate Last, Next_Last, Next_Next_Last */
		Last = ((Iter+1) == 1); NextLast = ((Iter+2) == 1); NextNextLast = ((Iter+3) == 1);
		/* Call Kernel LOC_INNER_LOOP */
		KerArg0->In = (Word16 * __restrict__) ((rt_pointerT) (PULP_Dronet_L1_Memory + 0) + 0);
		KerArg0->Filter = (Word16 * __restrict__) ((rt_pointerT) (PULP_Dronet_L1_Memory + 12548) + 0 + (0)*12544);
		KerArg0->Bias = (Word16 *  __restrict__) ((rt_pointerT) (PULP_Dronet_L1_Memory + 12544) + 0 + (0)*2);
		KerArg0->Out = (Word16 *  __restrict__) ((rt_pointerT) (PULP_Dronet_L1_Memory + 25092) + 0 + (0)*2);
		rt_team_fork(gap8_ncore(), (void *) KerLinearLayer_fp, (void *) KerArg0);
		N_Ti++;
		/* End Kernel Iteration Loop on Iter space */
	}
	Iter=1;
	/* =======================Write Last Tile=========================================== */
	rt_dma_memcpy((rt_pointerT) Out + (0),
		(rt_pointerT) (PULP_Dronet_L1_Memory + 25092) + 0, 2, RT_DMA_DIR_LOC2EXT, 0, &DmaW_Evt1);
	rt_dma_wait(&DmaW_Evt1);
	/* ===================End Write Last Tile=========================================== */
}

void LinearLayer_SW_2(
		Word16 * __restrict__ In,
		Word16 * __restrict__ Filter,
		unsigned int NormFilter,
		Word16 * __restrict__ Bias,
		unsigned int NormBias,
		Word16 * __restrict__ Out,
		int OutSize,
		Kernel_T *Ker)

{
	/* Local variables used by this kernel */
	rt_dma_copy_t DmaR_Evt1;
	rt_dma_copy_t DmaR_Evt2;
	rt_dma_copy_t DmaR_Evt3;
	rt_dma_copy_t DmaW_Evt1;
	int Iter;
	int Last, NextLast, NextNextLast;
	int N_Ti = 0;
	int N_TiIp = 0, InPlane, OutPlane=0;
	KerLinearLayer_fpT S_KerArg0, *KerArg0 = &S_KerArg0;

	/* Initialize KerArg, Kernel invariant arguments */
	KerArg0->InSize = (int) (6272);
	KerArg0->NormFilter = (NormFilter);
	KerArg0->NormBias = (NormBias);
	KerArg0->OutSize = (int) (1);
	/* =======================Read First Tile=========================================== */
	/* Initial reads in L2, O_DB or O_BUFF */
	rt_dma_memcpy((rt_pointerT) In+0, (rt_pointerT) (PULP_Dronet_L1_Memory + 0)+0, 12544, RT_DMA_DIR_EXT2LOC, 0, &DmaR_Evt1);
	rt_dma_memcpy((rt_pointerT) Bias+(0), (rt_pointerT) (PULP_Dronet_L1_Memory + 12544)+0, 2, RT_DMA_DIR_EXT2LOC, 0, &DmaR_Evt2);
	rt_dma_memcpy((rt_pointerT) Filter+(0), (rt_pointerT) (PULP_Dronet_L1_Memory + 12548)+0, 12544, RT_DMA_DIR_EXT2LOC, 0, &DmaR_Evt3);
	/* Wait for BUFF read in L2 */
	rt_dma_wait(&DmaR_Evt1);
	rt_dma_wait(&DmaR_Evt2);
	rt_dma_wait(&DmaR_Evt3);
	/* ===================End Read First Tile=========================================== */
	/* Kernel Iteration Loop on Iter space */
	Iter=0; {
		/* Loop Iteration Body on Iter space */
		/* Elaborate Last, Next_Last, Next_Next_Last */
		Last = ((Iter+1) == 1); NextLast = ((Iter+2) == 1); NextNextLast = ((Iter+3) == 1);
		/* Call Kernel LOC_INNER_LOOP */
		KerArg0->In = (Word16 * __restrict__) ((rt_pointerT) (PULP_Dronet_L1_Memory + 0) + 0);
		KerArg0->Filter = (Word16 * __restrict__) ((rt_pointerT) (PULP_Dronet_L1_Memory + 12548) + 0 + (0)*12544);
		KerArg0->Bias = (Word16 *  __restrict__) ((rt_pointerT) (PULP_Dronet_L1_Memory + 12544) + 0 + (0)*2);
		KerArg0->Out = (Word16 *  __restrict__) ((rt_pointerT) (PULP_Dronet_L1_Memory + 25092) + 0 + (0)*2);
		rt_team_fork(gap8_ncore(), (void *) KerLinearLayer_fp, (void *) KerArg0);
		N_Ti++;
		/* End Kernel Iteration Loop on Iter space */
	}
	Iter=1;
	/* =======================Write Last Tile=========================================== */
	rt_dma_memcpy((rt_pointerT) Out + (0),
		(rt_pointerT) (PULP_Dronet_L1_Memory + 25092) + 0, 2, RT_DMA_DIR_LOC2EXT, 0, &DmaW_Evt1);
	rt_dma_wait(&DmaW_Evt1);
	/* ===================End Write Last Tile=========================================== */
}

