/*
 * pulp_nn_depthwise_generic.c
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
#define SumDotp(a, b, c) __builtin_pulp_sdotusp4(a, b, c)
#define MIN(a,b) ((a)<(b)?(a):(b))
#define clip8(x) __builtin_pulp_clipu_r(x, 255)

void pulp_nn_depthwise_generic_less_4_weights(
  const uint8_t * Im_in,
  const uint16_t  dim_im_in_x,
  const uint16_t  dim_im_in_y,
  const uint16_t  ch_im_in,
  const int8_t *  wt,
  const uint16_t  ch_im_out,
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
  uint16_t        out_shift,
  uint16_t        out_mult,
  uint8_t *       Im_out,
  const uint16_t  dim_im_out_x,
  const uint16_t  dim_im_out_y,
  int64_t *       k,
  int64_t *       lambda,
  uint8_t *       bufferC,
  uint8_t *       bufferB,
  int8_t          FLAG_RELU,
  int8_t          FLAG_BATCH_NORM,
  unsigned int * memory_chan
){
  int core_id = pi_core_id();
  int chunk = (ch_im_out >> log2(NUM_CORES)) + ((ch_im_out & (NUM_CORES - 1)) != 0);
  int start_channel = MIN(chunk * core_id, ch_im_out);
  int stop_channel = MIN(start_channel + chunk, ch_im_out);
  int dim_kernel_x_size_padded = (dim_kernel_x >> 2) + ((dim_kernel_x & 0x3) != 0);
  int dim_incr = (dim_kernel_x_size_padded << 2) - dim_kernel_x;
  // + dim_kernel_x is a value to guard. It is necessary when there is asymetrical padding
  // uint8_t * bufferA = bufferC + (core_id * ((dim_kernel_x * (dim_im_in_y + padding_y_top + padding_y_bottom)) + dim_incr));
  uint8_t * bufferA = bufferC + (core_id * ((dim_kernel_x * (dim_im_in_y + padding_y_top + padding_y_bottom)) + dim_kernel_x + dim_incr));
  int i_out_x;
  int i_buff_y;
  uint8_t colCnt = (dim_kernel_y * dim_kernel_x) >> 2;
  uint8_t leftCnt = (dim_kernel_y * dim_kernel_x) & 0x3;

  for(int i_out_ch = start_channel; i_out_ch < stop_channel; i_out_ch++)
  {
    unsigned int i_ch_im = (i_out_ch * dim_im_in_x * dim_im_in_y);
    unsigned int i_ch_ker = (i_out_ch * dim_kernel_y * dim_kernel_x);
    i_out_x = 0;
    if(padding_x_left > 0)
    {
      do
      {
        uint8_t *pOut = Im_out + i_out_ch + (i_out_x * ch_im_out);
        uint8_t *pBuffer = bufferA;

        i_buff_y = - padding_y_top;
        if(padding_y_top > 0)
        {
          do
          {
            int i = 0;
            do
            {
              *(v4u *) pBuffer = (v4u) {0, 0, 0, 0};
              pBuffer+=4;
              i++;
            }while(i<dim_kernel_x_size_padded);
            pBuffer-=dim_incr;
            i_buff_y++;
          }while(i_buff_y < 0);
        }
        int const1 = padding_x_left - (i_out_x * stride_x);
        int base_ptr = Im_in + i_ch_im;
        do
        {
          for(int j=0; j<const1; j++)
          {
            *(uint8_t *) pBuffer = 0;
            pBuffer++;
          }
          int idx = 0;
          int i = 0;
          do
          {
            *((v4u*) pBuffer) = *((v4u*) (base_ptr + idx));
            pBuffer+=4;
            idx+=4;
            i++;
          }while(i<dim_kernel_x_size_padded);
          pBuffer-=(dim_incr + const1);
          base_ptr+=dim_im_in_x;
          i_buff_y++;
        }while(i_buff_y < dim_im_in_y);
        for(i_buff_y; i_buff_y < dim_im_in_y + padding_y_bottom; i_buff_y++)
        {
          int i = 0;
          do
          {
            *(v4u *) pBuffer = (v4u) {0, 0, 0, 0};
            pBuffer+=4;
            i++;
          }while(i<dim_kernel_x_size_padded);
          pBuffer-=dim_incr;
        }
        int l = 0;
        do
        {
          int8_t *pW = wt + i_ch_ker;

          int sum = 0;

          if (bias != NULL)
          {
            sum = ((int)(bias[i_out_ch]));
          }

          pBuffer = bufferA + (l * stride_y * dim_kernel_x);
          int j=0;
          asm volatile("":::"memory");
          for(int j=0; j<leftCnt; j++)
          {
            int8_t w = *(int8_t *) pW;
            uint8_t x = *(uint8_t *) pBuffer;
            asm volatile("":::"memory");
            sum += x * w;
            pW++;
            pBuffer++;
          }
          while(j<colCnt)
          {
            v4s w = *(v4s *) pW;
            v4u x = *(v4u *) pBuffer;
            asm volatile("":::"memory");
            sum  = SumDotp(x, w, sum);
            pBuffer += 4;
            pW += 4;
            j++;
          }
          if (FLAG_BATCH_NORM && FLAG_RELU)
          {
            *pOut = pulp_nn_bn_quant_u8(sum, *(k + start_channel), *(lambda + start_channel), out_shift);
          }
          else if (FLAG_RELU)
          {
            *pOut = pulp_nn_quant_u8(sum, out_mult, out_shift);
          }
          else
          {
            *pOut = (uint8_t) clip8(sum >> out_shift);
          }
          pOut+=(dim_im_out_x * ch_im_out);
          l++;
        }while(l<dim_im_out_y);
        i_out_x++;
      }while((i_out_x * stride_x) < padding_x_left);
    }
    do
    {
      uint8_t *pOut = Im_out + i_out_ch + (i_out_x * ch_im_out);
      uint8_t *pBuffer = bufferA;
      i_buff_y = - padding_y_top;
      if(padding_y_top > 0)
      {
        do
        {
          int i = 0;
          do
          {
            *(v4u *) pBuffer = (v4u) {0, 0, 0, 0};
            pBuffer+=4;
            i++;
          }while(i<dim_kernel_x_size_padded);
          pBuffer-=dim_incr;
          i_buff_y++;
        }while(i_buff_y < 0);
      }
      int base_ptr = Im_in + i_ch_im +  (i_out_x * stride_x) - padding_x_left;
      do
      {
        int idx = 0;
        for (int i=0; i<dim_kernel_x_size_padded; i++)
        {
          *((v4u*) pBuffer) = *((v4u*) (base_ptr + idx));
          idx+=4;
          pBuffer+=4;
        }
        pBuffer-=dim_incr;
        base_ptr+=dim_im_in_x;
        i_buff_y++;
      }while(i_buff_y < dim_im_in_y);
      for(i_buff_y; i_buff_y < dim_im_in_y + padding_y_bottom; i_buff_y++)
      {
        int i = 0;
        do
        {
          *(v4u *) pBuffer = (v4u) {0, 0, 0, 0};
          pBuffer+=4;
          i++;
        }while(i<dim_kernel_x_size_padded);
        pBuffer-=dim_incr;
      }
      int l = 0;
      do
      {
        int8_t *pW = wt + i_ch_ker;

        int sum = 0;

        if (bias != NULL)
        {
          sum = ((int)(bias[i_out_ch]));
        }

        pBuffer = bufferA + (l * stride_y * dim_kernel_x);
        int j=0;
        asm volatile("":::"memory");
        for(int j=0; j<leftCnt; j++)
        {
          int8_t w = *(int8_t *) pW;
          uint8_t x = *(uint8_t *) pBuffer;
          asm volatile("":::"memory");
          sum += x * w;
          pW++;
          pBuffer++;
        }
        while(j<colCnt)
        {
          v4s w = *(v4s *) pW;
          v4u x = *(v4u *) pBuffer;
          asm volatile("":::"memory");
          sum  = SumDotp(x, w, sum);
          pBuffer += 4;
          pW += 4;
          j++;
        }
        if (FLAG_BATCH_NORM && FLAG_RELU)
        {
          *pOut = pulp_nn_bn_quant_u8(sum, *(k + start_channel), *(lambda + start_channel), out_shift);
        }
        else if (FLAG_RELU)
        {
          *pOut = pulp_nn_quant_u8(sum, out_mult, out_shift);
        }
        else
        {
          *pOut = (uint8_t) clip8(sum >> out_shift);
        }
        pOut+=(dim_im_out_x * ch_im_out);
        l++;
      }while(l<dim_im_out_y);
      i_out_x++;
    }while((i_out_x * stride_x) < ((dim_im_out_x * stride_x) - padding_x_right));
    for (i_out_x; i_out_x < dim_im_out_x; i_out_x++)
    {
      uint8_t *pOut = Im_out + i_out_ch + (i_out_x * ch_im_out);
      uint8_t *pBuffer = bufferA;
      i_buff_y = - padding_y_top;
      if(padding_y_top > 0)
      {
        do
        {
          int i = 0;
          do
          {
            *(v4u *) pBuffer = (v4u) {0, 0, 0, 0};
            pBuffer+=4;
            i++;
          }while(i<dim_kernel_x_size_padded);
          pBuffer-=dim_incr;
          i_buff_y++;
        }while(i_buff_y < 0);
      }
      int base_ptr = Im_in + i_ch_im + (i_out_x * stride_x) - padding_x_left;
      do
      {
        int i = 0;
        int idx = 0;
        do
        {
          *((v4u*) pBuffer) = *((v4u*) (base_ptr + idx));
          idx+=4;
          pBuffer+=4;
          i++;
        }while(i<dim_kernel_x_size_padded);
        base_ptr+=dim_im_in_x;
        pBuffer-=(dim_incr + 1 + (i_out_x * stride_x) - (dim_im_out_x * stride_x) + padding_x_right);
        for(int j=0; j<(1 + (i_out_x * stride_x) - (dim_im_out_x * stride_x) + padding_x_right); j++)
        {
          *(uint8_t *) pBuffer = 0;
          pBuffer++;
        }
        i_buff_y++;
      }while(i_buff_y < dim_im_in_y);
      for(i_buff_y; i_buff_y < dim_im_in_y + padding_y_bottom; i_buff_y++)
      {
        int i = 0;
        do
        {
          *(v4u *) pBuffer = (v4u) {0, 0, 0, 0};
          pBuffer+=4;
          i++;
        }while(i<dim_kernel_x_size_padded);
        pBuffer-=dim_incr;
      }
      int l = 0;
      do
      {
        int8_t *pW = wt + i_ch_ker;

        int sum = 0;

        if (bias != NULL)
        {
          sum = ((int)(bias[i_out_ch]));
        }

        pBuffer = bufferA + (l * stride_y * dim_kernel_x);
        int j=0;
        asm volatile("":::"memory");
        for(int j=0; j<leftCnt; j++)
        {
          int8_t w = *(int8_t *) pW;
          uint8_t x = *(uint8_t *) pBuffer;
          asm volatile("":::"memory");
          sum += x * w;
          pW++;
          pBuffer++;
        }
        while(j<colCnt)
        {
          v4s w = *(v4s *) pW;
          v4u x = *(v4u *) pBuffer;
          asm volatile("":::"memory");
          sum  = SumDotp(x, w, sum);
          pBuffer += 4;
          pW += 4;
          j++;
        }
        if (FLAG_BATCH_NORM && FLAG_RELU)
        {
          *pOut = pulp_nn_bn_quant_u8(sum, *(k + start_channel), *(lambda + start_channel), out_shift);
        }
        else if (FLAG_RELU)
        {
          *pOut = pulp_nn_quant_u8(sum, out_mult, out_shift);
        }
        else
        {
          *pOut = (uint8_t) clip8(sum >> out_shift);
        }
        pOut+=(dim_im_out_x * ch_im_out);
        l++;
      }while(l<dim_im_out_y);
    }
    k++;
    lambda++;
  }
  pi_cl_team_barrier(0);
}
