/*
 * pulp_nn_avgpool.c
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
#define clip8(x) __builtin_pulp_clipu_r(x, 255)

void __attribute__ ((noinline))  pulp_nn_avgpool (
  uint8_t *  Im_in,            // pointer to the input feature map
  uint16_t  dim_im_in_x,      // spatial dimension of the input feature map
  uint16_t  dim_im_in_y,
  uint16_t  ch_im_in,         // number of channels of the IFM
  uint16_t  dim_kernel_x,       // spatial dimension of the pooling filter
  uint16_t  dim_kernel_y,       // spatial dimension of the pooling filter
  uint16_t  padding,          // amount of padding
  uint16_t  stride,           // amount of stride
  uint16_t  dim_im_out_x,     // reduced spatial dimension of output
  uint16_t  dim_im_out_y,
  int8_t *  bufferA,          // actually not used in this fx
  uint8_t *  Im_out,           // pointer to the output
  int32_t * pOutBufferAcc,
  int8_t    flag_acc_buff_out,
  int8_t    flag_first_ch_out,
  int       flag_relu,
  const uint16_t  out_shift,
  const uint16_t  out_mult
) {
  int core_id = pi_core_id();
  int n_cores = NUM_CORES;
  int i,j;
  if (pi_core_id()==0)
  {
    for (int scan_channel = 0; scan_channel<ch_im_in; scan_channel++)
    {
      j=0;
      for (int h_in = 0; h_in<dim_im_in_y-dim_kernel_y+1; h_in+=stride)
      {
        i=0;
        for (int w_in = 0; w_in<dim_im_in_x-dim_kernel_x+1; w_in+=stride)
        {
          uint32_t sum=0;
          for (int h_out=0;h_out<dim_kernel_y;h_out++){
            for (int w_out=0;w_out<dim_kernel_x;w_out++){
              sum += (uint32_t)*(Im_in+scan_channel + ch_im_in*(w_in+w_out)+ch_im_in*dim_im_in_x*(h_in+h_out));
            }
          }
          if (flag_relu){
            sum = (sum * out_mult) >> out_shift;
            sum = sum/(dim_kernel_x*dim_kernel_y);
            sum = clip8(sum);}
          else
            sum = sum/(dim_kernel_x*dim_kernel_y);
          *(Im_out+scan_channel + ch_im_in*(w_in)+ch_im_in*dim_im_in_x*(h_in))=(uint8_t)sum;
          i++;
        }
        j++;
      }
    }
  }
  pi_cl_team_barrier(0);

}
