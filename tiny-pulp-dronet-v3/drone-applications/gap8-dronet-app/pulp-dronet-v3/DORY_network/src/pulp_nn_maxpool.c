/*
 * pulp_nn_maxpool.c
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

void __attribute__ ((noinline))  pulp_nn_maxpool (
	uint8_t * Im_in,             // pointer to the input feature map
	uint16_t  dim_im_in_x,       // spatial dimension of the input feature map
	uint16_t  dim_im_in_y,
	uint16_t  ch_im_in,          // number of channels of the IFM
	uint16_t  dim_kernel_x,        // spatial dimension of the pooling filter
	uint16_t  dim_kernel_y,        // spatial dimension of the pooling filter
	uint16_t  padding_t,           // amount of padding
	uint16_t  padding_b,           // amount of padding
	uint16_t  padding_l,           // amount of padding
	uint16_t  padding_r,           // amount of padding
	uint16_t  stride,            // amount of stride
	uint16_t  dim_im_out_x,      // reduced spatial dimension of output
	uint16_t  dim_im_out_y,
	int8_t *  bufferA,           // actually not used in this fx
	uint8_t * Im_out,            // pointer to the output
	int32_t * pOutBufferAcc,
	int8_t    flag_acc_buff_out,
	int8_t    flag_first_ch_out
) {
	int core_id = pi_core_id();
	int n_cores = NUM_CORES;
	if (dim_im_in_y < NUM_CORES)
	{
	  n_cores = dim_im_in_y;
	}
	int  Log2Core = log2(n_cores);

  int chunck = (dim_im_in_y >> Log2Core) + ((dim_im_in_y & (NUM_CORES-1))!=0);

  int start = min(chunck * core_id, dim_im_in_y);
  int stop = min(start + chunck, dim_im_in_y);
  int16_t   i_x, i_y;

	for (i_y = start; i_y < stop; i_y++)
	{
		for (i_x = 0; i_x < dim_im_out_x; i_x++)
		{
			/* for each output pixel */
			uint8_t     *target = Im_in + (i_y * dim_im_in_x + i_x) * ch_im_in; //to test: prob dim_im_in_x
			uint8_t     *win_start;
			uint8_t     *win_stop;
			if (i_x * stride - padding_l < 0)
			{
				win_start = target;
			}
			else
			{
				win_start = Im_in + (i_y * dim_im_in_x + i_x * stride - padding_l) * ch_im_in;//to test: prob dim_im_in_x
			}

			if (i_x * stride - padding_l + dim_kernel_x >= dim_im_in_x)
			{
				win_stop = Im_in + (i_y * dim_im_in_x + dim_im_in_x) * ch_im_in;//to test: prob dim_im_in_x
			}
			else
			{
				win_stop = Im_in + (i_y * dim_im_in_x + i_x * stride - padding_l + dim_kernel_x) * ch_im_in;//to test: prob dim_im_in_x
			}

			/* first step is to copy over initial data */
			for (int i = 0; i< ch_im_in; i++) target[i] = win_start[i];

			/* start the max operation from the second part */
			win_start += ch_im_in;
			for (; win_start < win_stop; win_start += ch_im_in)
			{
				pulp_nn_compare_and_replace_if_larger_int8(target, win_start, ch_im_in);
			}
		}
	}

	pi_cl_team_barrier(0);
  if (dim_im_out_y < NUM_CORES)
	{
    n_cores = dim_im_out_y;
	}
  Log2Core = log2(n_cores);
  int chunck2 = (dim_im_out_y >> Log2Core) + ((dim_im_out_y & (NUM_CORES-1))!=0);
  int start2 = chunck2 * core_id;//, dim_im_out_y);
  int stop2 = min(start2 + chunck2, dim_im_out_y);

	/* every core works on its part of accumulation buffer */
	int32_t *pBuffAcc = pOutBufferAcc + (start2 * dim_im_out_x);

	/* then does the pooling along y axis */
  for (i_y = start2; i_y < stop2; i_y++)
  {
    /* for each output row */
    uint8_t *target = Im_out + i_y * dim_im_out_x * ch_im_in; //to test: prob dim_im_out_x
    uint8_t *row_start;
    uint8_t *row_end;
    /* setting the starting row */
    if (i_y * stride - padding_t < 0)
    {
      row_start = Im_in;
    }
    else
    {
      row_start = Im_in + (i_y * stride - padding_t) * dim_im_in_x * ch_im_in; //to test: prob dim_im_in_x
    }
    /* setting the stopping row */
    if (i_y * stride - padding_t + dim_kernel_y >= dim_im_in_y)
    {
      row_end = Im_in + dim_im_in_x * dim_im_in_y * ch_im_in;//to test: prob dim_im_in_x
    }
    else
    {
     	row_end = Im_in + (i_y * stride - padding_t + dim_kernel_y) * dim_im_in_x * ch_im_in; //to test: prob dim_im_in_x
    }

    /* copy over the first row */
    for (int i = 0; i< dim_im_out_x * ch_im_in; i++)
		{
			target[i] = row_start[i];
		}
    /* move over to next row */
    row_start += ch_im_in * dim_im_in_x;

    for (; row_start < row_end; row_start += dim_im_in_x * ch_im_in)
    {
      pulp_nn_compare_and_replace_if_larger_int8(target, row_start, dim_im_out_x * ch_im_in);
			if(flag_acc_buff_out == 1)
			{
				if(flag_first_ch_out == 1)
				{
					*pBuffAcc = 0;
				}
				else
				{
					/* to test with builtin (sdotp4) */
					*pBuffAcc+=(target[0] + target[1] + target[2] + target[3]);
				}
			}
    }
		if(flag_acc_buff_out == 1)
		{
			pBuffAcc++;
		}
  }
 	pi_cl_team_barrier(0);
}
