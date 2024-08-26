/*
 * pulp_nn_add.c
 * Nazareno Bruschi <nazareno.bruschi@unibo.it>
 * Angelo Garofalo <angelo.garofalo@unibo.it>
 *
 * Copyright (C) 2018-2020 University of Bologna
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "pmsis.h"
#include "pulp_nn_utils.h"

#define log2(x) __builtin_pulp_fl1(x)
#define min(a,b) ((a)<(b)?(a):(b))

void __attribute__ ((noinline))  pulp_nn_add (
	uint8_t * Im_in_1,             // pointer to the input feature map1
	uint8_t * Im_in_2,             // pointer to the input feature map2
	uint16_t  ch_im_in,          // number of channels of the IFM
	uint16_t  dim_im_in_h,
	uint16_t  dim_im_in_w,
	uint8_t * Im_out,            // pointer to the output
	uint16_t out_mult1,            // paramter to requantize
	uint16_t out_mult2,            // paramter to requantize
	uint16_t out_shift            // paramter to requantize
)
{
	int core_id = pi_core_id();
	int n_cores = NUM_CORES;
	if (dim_im_in_h < NUM_CORES)
	{
	  n_cores = dim_im_in_h;
	}
	int  Log2Core = log2(n_cores);

	int chunck = (dim_im_in_h >> Log2Core) + ((dim_im_in_h & (NUM_CORES-1))!=0);

	int start = min(chunck * core_id, dim_im_in_h);
	int stop = min(start + chunck, dim_im_in_h);
	uint8_t *target1 = Im_in_1 + start*ch_im_in*dim_im_in_w;
	uint8_t *target2 = Im_in_2 + start*ch_im_in*dim_im_in_w;
	uint8_t *pOut = Im_out + start*ch_im_in*dim_im_in_w;
	uint8_t res = 0;
	for (int spatial = 0; spatial<dim_im_in_w*ch_im_in*(stop-start); spatial+=1)
	{
		res = pulp_nn_add_quant_u8(*target1, *target2, out_mult1, out_mult2, out_shift);
		*pOut = res;
		target1 += 1;
		target2 += 1;
		pOut += 1;
	}
   pi_cl_team_barrier(0);
}
