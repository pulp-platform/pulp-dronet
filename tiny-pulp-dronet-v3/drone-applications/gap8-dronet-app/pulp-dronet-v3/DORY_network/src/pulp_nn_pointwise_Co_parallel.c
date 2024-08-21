/*
 * pulp_nn_pointwise_Co_parallel.c
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
#include "pulp_nn_kernels.h"

#define log2(x) __builtin_pulp_fl1(x)
#define min(a,b) ((a)<(b)?(a):(b))
#define SumDotp(a, b, c)        __builtin_pulp_sdotusp4(a, b, c)
#define clip8(x)                __builtin_pulp_clipu_r(x, 255)

void __attribute__ ((noinline)) pulp_nn_pointwise_Co_parallel(
  const uint8_t * pInBuffer,
  const uint16_t  dim_in_x,
  const uint16_t  dim_in_y,
  const uint16_t  ch_in,
  const int8_t *  pWeight,
  const uint16_t  ch_out,
  const uint16_t  dim_kernel_x,
  const uint16_t  dim_kernel_y,
  const uint16_t  padding_y_top,
  const uint16_t  padding_y_bottom,
  const uint16_t  padding_x_left,
  const uint16_t  padding_x_right,
  const uint16_t  stride_x,
  const uint16_t  stride_y,
  const int8_t *  bias,
  const uint16_t  bias_shift,
  const uint16_t  out_shift,
  const uint16_t  out_mult,
  uint8_t *       pOutBuffer,
  const uint16_t  dim_out_x,
  const uint16_t  dim_out_y,
  int64_t *       k,
  int64_t *       lambda,
  uint8_t *       pIm2ColBuffer,
  int             flag_relu,
  int             flag_batch_norm,
  unsigned int * memory_chan
) {
  int core_id = pi_core_id();

  // local vars
  int i_out_y, i_out_x, i_ker_y, i_ker_x;
  int Log2Core = log2(NUM_CORES);

  int chunk = (ch_out >> Log2Core) + ((ch_out & (NUM_CORES - 1)) != 0);

  /* defining the specific channels computed by each core */
  int start_channel, stop_channel;
  start_channel = min(chunk * core_id, ch_out);
  stop_channel = min(start_channel + chunk, ch_out);

  int eff_chunk = stop_channel - start_channel;

  int8_t *pW = pWeight + (start_channel * ch_in);

  int64_t *k0 = k + start_channel;
  int64_t *lambda0 = lambda + start_channel;

  if(eff_chunk)
  {
    for (i_out_y = 0; i_out_y < dim_out_y; i_out_y++)
    {
      i_out_x = 0;

      uint8_t *pOut = pOutBuffer + start_channel + (i_out_y * dim_out_x * ch_out);

      for (int n = 0; n < dim_out_x; n++)
      {
        uint8_t *pB = (pInBuffer + (i_out_x * ch_in) + (i_out_y * dim_in_x * ch_in));
        pOut = ((ch_out - eff_chunk) << 1) + pulp_nn_matmul(
          pW,
          pB,
          eff_chunk,
          ch_in,
          bias_shift,
          out_shift,
          out_mult,
          k0,
          lambda0,
          bias,
          pOut,
          pOut + ch_out,
          flag_relu,
          flag_batch_norm
        );
        i_out_x+=2;
      }
      /* check if there is left-over for compute */
      if (i_out_x != dim_out_x)
      {
        const int8_t *pA = pWeight;

        for (int i = start_channel; i < stop_channel; i++)
        {
          int sum = 0;

          if (bias != NULL)
          {
            sum = ((int)(bias[i]));
          }

          uint8_t *pB = (pInBuffer + (i_out_x * ch_in) + (i_out_y * dim_in_x * ch_in));;
          /* basically each time it process 4 entries */
          uint16_t  col_cnt_im2col = ch_in >> 2;

          for (int j=0 ; j < col_cnt_im2col; j++)
          {
            v4s inA = *((v4s*) pA);
            v4u inB = *((v4u*) pB);

            sum = SumDotp(inB, inA, sum);
            pA+=4;
            pB+=4;
          }
          col_cnt_im2col = ch_in & 0x3;
          while (col_cnt_im2col)
          {
            int8_t      inA1 = *pA++;
            uint8_t     inB1 = *pB++;
            asm volatile("": : :"memory");
            sum += inA1 * inB1;

            col_cnt_im2col--;
          }
          /* if activation layer follows batch normalization */
          if (flag_batch_norm && flag_relu)
          {
            *pOut = pulp_nn_bn_quant_u8(sum, *k0, *lambda0, out_shift);
            k0++;
            lambda0++;
            pOut++;
          }
          else
          {
            /* if there isn't batch normalization but there is activation layer */
            if(flag_relu == 1)
            {
              *pOut = pulp_nn_quant_u8(sum, out_mult, out_shift);
            }
            else
            {
              *pOut = (uint8_t) clip8(sum >> out_shift);
            }
            pOut++;
          }
        }
      }
    }
  }
  pi_cl_team_barrier(0);
}
