/*
 * pulp_nn_matmul.c
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

#define SumDotp(a, b, c)    __builtin_pulp_sdotusp4(a, b, c)
#define clip8(x)            __builtin_pulp_clipu_r(x, 255)

uint8_t __attribute__ ((noinline)) *pulp_nn_matmul(
  const int8_t * pWeight,
  uint8_t *      pInBuffer,
  uint16_t       ch_out,
  uint16_t       num_col_im2col,
  uint16_t       bias_shift,
  uint16_t       out_shift,
  uint16_t       out_mult,
  int64_t *      k,
  int64_t *      lambda,
  const int8_t * bias,
  uint8_t *      pOut,
  uint8_t *      pOut2,
  int            flag_relu,
  int            flag_batch_norm
) {
  int8_t  *pA = pWeight;
  uint16_t chan_left = ch_out & 0x3;

  v4s vecA;
  v4s vecA2;
  v4s vecA3;
  v4s vecA4;
  v4u vecB;
  v4u vecB2;

  /* this loop over the OFM channels */
  for (int i = 0; i < ch_out>>2; i++)
  {
    uint8_t *pB  =  pInBuffer ;
    uint8_t *pB2 = (pB + num_col_im2col);
    int8_t *pA2 = (pA + num_col_im2col);
    int8_t *pA3 = (pA2 + num_col_im2col);
    int8_t *pA4 = (pA3 + num_col_im2col);

    int bias1 = 0;
    int bias2 = 0;
    int bias3 = 0;
    int bias4 = 0;
    int bias5 = 0;
    int bias6 = 0;
    int bias7 = 0;
    int bias8 = 0;

    if(bias != NULL)
    {
      bias1 = ((int) (*bias++));
      bias2 = ((int) (*bias++));
      bias3 = ((int) (*bias++));
      bias4 = ((int) (*bias++));

      bias5 = bias1;
      bias6 = bias2;
      bias7 = bias3;
      bias8 = bias4;
    }

    /* init the accumulators with corresponding biases. if there is the accumulation buffer, i included it here */
    int sum =  bias1;
    int sum2 = bias2;
    int sum3 = bias3;
    int sum4 = bias4;

    int sum5 = bias5;
    int sum6 = bias6;
    int sum7 = bias7;
    int sum8 = bias8;

    uint16_t  col_cnt_im2col = num_col_im2col & 0x3;

    for (int j=0; j < num_col_im2col >> 2 ; j++)
    {
      vecA  = * ( (v4s*) pA  );
      vecA2 = * ( (v4s*) pA2 );
      vecA3 = * ( (v4s*) pA3 );
      vecA4 = * ( (v4s*) pA4 );
      vecB  = * ( (v4u*) pB  );
      vecB2 = * ( (v4u*) pB2 );

      sum  =  SumDotp (vecB,  vecA,  sum  );
      sum2 =  SumDotp (vecB,  vecA2, sum2 );
      sum3 =  SumDotp (vecB,  vecA3, sum3 );
      sum4 =  SumDotp (vecB,  vecA4, sum4 );

      sum5 =  SumDotp (vecB2, vecA,  sum5 );
      sum6 =  SumDotp (vecB2, vecA2, sum6 );
      sum7 =  SumDotp (vecB2, vecA3, sum7 );
      sum8 =  SumDotp (vecB2, vecA4, sum8 );

      pA  += 4;
      pA2 += 4;
      pA3 += 4;
      pA4 += 4;
      pB  += 4;
      pB2 += 4;
    }

    while(col_cnt_im2col)
    {
      int8_t      inA  = *pA++;
      int8_t      inA2 = *pA2++;
      int8_t      inA3 = *pA3++;
      int8_t      inA4 = *pA4++;
      uint8_t     inB  = *pB++;
      uint8_t     inB2 = *pB2++;
      asm volatile("": : :"memory");
      sum  += inA  * inB;
      sum2 += inA2 * inB;
      sum3 += inA3 * inB;
      sum4 += inA4 * inB;
      sum5 +=  inA * inB2;
      sum6 += inA2 * inB2;
      sum7 += inA3 * inB2;
      sum8 += inA4 * inB2;

      col_cnt_im2col--;
    }

    if (flag_batch_norm && flag_relu)
    {
      *pOut = pulp_nn_bn_quant_u8(sum, *k, *lambda, out_shift);
      pOut++;
      *pOut2 = pulp_nn_bn_quant_u8(sum5, *k, *lambda, out_shift);
      pOut2++;
      k++;
      lambda++;

      *pOut = pulp_nn_bn_quant_u8(sum2, *k, *lambda, out_shift);
      pOut++;
      *pOut2 = pulp_nn_bn_quant_u8(sum6, *k, *lambda, out_shift);
      pOut2++;
      k++;
      lambda++;

      *pOut = pulp_nn_bn_quant_u8(sum3, *k, *lambda, out_shift);
      pOut++;
      *pOut2 = pulp_nn_bn_quant_u8(sum7, *k, *lambda, out_shift);
      pOut2++;
      k++;
      lambda++;

      *pOut = pulp_nn_bn_quant_u8(sum4, *k, *lambda, out_shift);
      pOut++;
      *pOut2 = pulp_nn_bn_quant_u8(sum8, *k, *lambda, out_shift);
      pOut2++;
      k++;
      lambda++;
    }
    else
    {
      if (flag_relu == 1)
      {
        *pOut = pulp_nn_quant_u8(sum, out_mult, out_shift);
        pOut++;
        *pOut = pulp_nn_quant_u8(sum2, out_mult, out_shift);
        pOut++;
        *pOut = pulp_nn_quant_u8(sum3, out_mult, out_shift);
        pOut++;
        *pOut = pulp_nn_quant_u8(sum4, out_mult, out_shift);
        pOut++;

        *pOut2 = pulp_nn_quant_u8(sum5, out_mult, out_shift);
        pOut2++;
        *pOut2 = pulp_nn_quant_u8(sum6, out_mult, out_shift);
        pOut2++;
        *pOut2 = pulp_nn_quant_u8(sum7, out_mult, out_shift);
        pOut2++;
        *pOut2 = pulp_nn_quant_u8(sum8, out_mult, out_shift);
        pOut2++;
      }
      else
      {
        *pOut = (uint8_t) clip8(sum >> out_shift);
        pOut++;
        *pOut = (uint8_t) clip8(sum2 >> out_shift);
        pOut++;
        *pOut = (uint8_t) clip8(sum3 >> out_shift);
        pOut++;
        *pOut = (uint8_t) clip8(sum4 >> out_shift);
        pOut++;

        *pOut2 = (uint8_t) clip8(sum5 >> out_shift);
        pOut2++;
        *pOut2 = (uint8_t) clip8(sum6 >> out_shift);
        pOut2++;
        *pOut2 = (uint8_t) clip8(sum7 >> out_shift);
        pOut2++;
        *pOut2 = (uint8_t) clip8(sum8 >> out_shift);
        pOut2++;
      }
    }

    pA +=  3 * num_col_im2col;
  }

  while(chan_left)
  {
    uint8_t *pB  =  pInBuffer ;
    uint8_t *pB2 = (pB + num_col_im2col);

    int bias1 = 0;
    int bias2 = 0;

    if (bias != NULL)
    {
      bias1 = ((int) (*bias++));
      bias2 = bias1;
    }

    int sum  =  bias1;
    int sum2 =  bias2;

    for (int j=0; j < num_col_im2col >> 2 ; j++)
    {
      vecA  = * ( (v4s*) pA  );
      vecB  = * ( (v4u*) pB  );
      vecB2 = * ( (v4u*) pB2 );

      sum  =  SumDotp (vecB, vecA, sum  );
      sum2 =  SumDotp (vecB2,vecA, sum2 );

      pA  += 4;
      pB  += 4;
      pB2 += 4;
    }

    uint16_t  col_cnt_im2col =num_col_im2col & 0x3;

    while(col_cnt_im2col)
    {
      int8_t      inA  = *pA++;
      uint8_t     inB  = *pB++;
      uint8_t     inB2 = *pB2++;
      asm volatile("": : :"memory");
      sum  += inA  * inB;
      sum2 +=  inA * inB2;

      col_cnt_im2col--;
    }

    if (flag_batch_norm && flag_relu)
    {
      *pOut = pulp_nn_bn_quant_u8(sum, *k, *lambda, out_shift);
      pOut++;
      *pOut2 = pulp_nn_bn_quant_u8(sum2, *k, *lambda, out_shift);
      pOut2++;
      k++;
      lambda++;
    }
    else
    {
      if (flag_relu == 1)
      {
        *pOut = pulp_nn_quant_u8(sum, out_mult, out_shift);
        pOut++;

        *pOut2 = pulp_nn_quant_u8(sum2, out_mult, out_shift);
        pOut2++;
      }
      else
      {
        *pOut = (uint8_t) clip8(sum >> out_shift);
        pOut++;

        *pOut2 = (uint8_t) clip8(sum2 >> out_shift);
        pOut2++;
      }
    }
    chan_left--;
  }
  pOut += ch_out;
  return pOut;
}
