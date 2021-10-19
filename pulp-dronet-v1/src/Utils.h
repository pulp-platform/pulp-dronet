/*----------------------------------------------------------------------------*
 * Copyright (C) 2019 ETH Zurich, Switzerland                                 *
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
 * File:    Utils.h                                                           *
 * Author:  Daniele Palossi <dpalossi@iis.ee.ethz.ch>                         *
 * Date:    12.12.2019                                                        *
 *----------------------------------------------------------------------------*/


#define ABS(x)   ( (x<0) ? (-x) : x ) 
#define MAX(x,y) ( (x>y) ? x : y )
#define MIN(x,y) ( (x<y) ? x : y )

#ifdef DEBUG

 /* Conversion function: from fixed-point to float
 * x: 16 bit input fixed-point number
 * qf: quantization factor*/
static float fixed2float(signed short int x, unsigned int qf) {

	float acc = 0.5f;

	// equivalent to powf(2, -qf)
	for(int i=0; i<qf-1; i++)
		acc *= 0.5f;

	return acc*x;
}


 /* Conversion function: from float to fixed-point
 * x: 32 bit input float number
 * qf: quantization factor*/
static signed short int float2fixed(float x, unsigned int qf) {

	return (signed short int)(x * (1 << qf));
}


 /* Dump Feature Maps function.
 * id: layer id to be dumped
 * FM: pointer to the feature maps
 * in_out: select to dump input FMs (0) or output FMs (1)
 * type: select to print as fixed-point (0) or float (1) */
static void dumpFMs(int id, short int *FM, int in_out, int type) {

	int channels	= 0;
	int size		= 0;
	int Norm		= NORM_ACT;
	int width		= 0;
	int height		= 0;

	if(in_out==0) {	// Input
		channels	= inCh[id];
		width		= inW[id];
		height		= inH[id];
		// We always use Q11 for FMs except for the Input Q8
		if(id==0) Norm = NORM_INPUT;
	} else {		// Output
		channels	= outCh[id];
		width		= outW[id];
		height		= outH[id];
	}

#ifdef VERBOSE
	printf("== layer %d ==\n", id);
#endif

	for(int i=0; i<channels; i++) {
		for(int j=0; j<height; j++) {
			for(int k=0; k<width; k++) {
				if(type==0)
					printf("0x%04x\n", (unsigned short int) FM[i*width*height+j*width+k]);
				else
					printf("%f\n", fixed2float(FM[i*width*height+j*width+k], Norm));
			}
		}
	}
}


/* Dump Weights for Conv layers.
 * id: layer id to be dumped
 * W: pointer to the weights
 * type: select to print as fixed-point (0) or float (1) */
static void dumpW(int id, short int *W, int type) {

	int ich		= inCh[id];
	int och		= outCh[id];
	int kWidth	= kerW[LAYERS_MAPPING_LUT[id]];
	int kHeight	= kerH[LAYERS_MAPPING_LUT[id]];
	int qFact	= Q_Factor[LAYERS_MAPPING_LUT[id]];

#ifdef VERBOSE
	printf("== layer %d ==\n", id);
#endif

	for(int i=0; i<ich; i++) {
		for(int j=0; j<och; j++) {
			for(int k=0; k<kHeight; k++) {
				for(int h=0; h<kWidth; h++) {
					if(type==0)
						printf("0x%04x\n", (unsigned short int) W[i*och*kHeight*kWidth + j*kHeight*kWidth + k*kWidth + h]);
					else 
						printf("%f\n", fixed2float(W[i*och*kHeight*kWidth + j*kHeight*kWidth + k*kWidth + h], qFact));
				}
			}
		}
	}
}


/* Dump Bias for Conv layers.
 * id: layer id to be dumped
 * bias: pointer to the bias
 * type: select to print as fixed-point (0) or float (1) */
static void dumpBias(int id, short int *bias, int type) {

	int och = outCh[id];

#ifdef VERBOSE
	printf("== layer %d ==\n", id);
#endif

	for(int i=0; i<och; i++) {
		if(type==0)
			printf("0x%04x\n", (unsigned short int) bias[i]);
		else
			printf("%f\n", fixed2float(bias[i], NORM_ACT));
	}
}

#endif // DEBUG