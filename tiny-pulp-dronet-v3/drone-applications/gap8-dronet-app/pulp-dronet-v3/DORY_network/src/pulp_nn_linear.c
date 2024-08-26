/*
 * pulp_nn_linear.c
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
#define SumDotp(a, b, c) __builtin_pulp_sdotusp4(a, b, c)
#define clip8(x) __builtin_pulp_clipu_r(x, 255)

void pulp_nn_linear(
      uint8_t *pInBuffer,
      int8_t *pWeights,
      uint16_t dim_vec,
      uint16_t num_o_neurons,
      int8_t *bias,
      uint16_t bias_shift,
      int8_t out_shift,
      uint16_t out_mult,
      int64_t *k,
      int64_t *lambda,
      uint8_t *pOutBuffer,
      int flag_relu,
      int flag_batch_norm,
      unsigned int * memory_chan
)
{
  int core_id = pi_core_id();
  int Log2Core = log2(NUM_CORES);
  int chunk = (num_o_neurons >> Log2Core) + ((num_o_neurons & (NUM_CORES-1))!=0);
  int start = min(chunk * core_id, num_o_neurons);
  int stop = min(start + chunk, num_o_neurons);

  v4u vecA;
  v4s vecB;

  uint8_t *pOut = (uint8_t *) pOutBuffer + start;

  int64_t *k1 = k + start;
  int64_t *lambda1 = lambda + start;

  for(int i=start; i<stop; i++)
  {
    int sum = 0;

    if (bias != NULL)
    {
      sum = ((int)(bias[i]));
    }

    uint8_t *pA = pInBuffer;
    int8_t *pB = pWeights + (i * dim_vec);

    for (int j=0; j<(dim_vec >> 2); j++)
    {
      vecA = *((v4u*)pA);
      vecB = *((v4s*)pB);
      sum = SumDotp(vecA, vecB, sum);
      pA+=4;
      pB+=4;
    }
    uint16_t col_cnt = dim_vec & 0x3;
    while (col_cnt)
    {
      uint8_t inA = *pA;
      pA++;
      int8_t inB = *pB;
      pB++;
      sum += inA * inB;
      col_cnt--;
    }
    if (flag_batch_norm && flag_relu)
    {
      *pOut = pulp_nn_bn_quant_u8(sum, *k1, *lambda1, out_shift);
      pOut++;
      k1++;
      lambda1++;
    }
    else
    {
      if (flag_relu == 1)
      {
        *pOut = pulp_nn_quant_u8(sum, out_mult, out_shift);
        pOut++;
      }
      else
      {
        *pOut = (uint8_t) clip8(sum >> out_shift);
        pOut++;
      }
    }
  }
  pi_cl_team_barrier(0);
}
