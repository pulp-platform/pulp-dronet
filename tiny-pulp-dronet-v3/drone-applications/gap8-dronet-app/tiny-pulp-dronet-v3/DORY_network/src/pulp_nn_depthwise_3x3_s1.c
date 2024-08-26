/*
 * pulp_nn_depthwise_3x3_s1.c
 * Nazareno Bruschi <nazareno.bruschi@unibo.it>
 * Francesco Conti <f.conti@unibo.it>
 *
 * Copyright (C) 2020 University of Bologna
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
#define clip8(x) __builtin_pulp_clipu(x, 0, 255)

void __attribute__ ((noinline)) pulp_nn_depthwise_3x3_s1(
      uint8_t * In_Img,
      uint8_t * Out_Img,
      int R,
      int C,
      int CH,
      int p_l,
      int p_r,
      int8_t  * Kernel,
      int8_t out_shift,
      uint16_t out_mult,
      int64_t * kappa,
      int64_t * lambda,
      int flag_relu,
      int flag_batch_norm,
      unsigned int * memory_chan
) { 
  uint8_t core_id = pi_core_id();
  uint8_t Log2Core = log2(NUM_CORES);

  int chunk = (CH >> Log2Core) + ((CH & (NUM_CORES - 1)) != 0);

  int lb = min(chunk * core_id, CH);
  int ub = min(lb + chunk, CH);

  int ch, r, c, t0, t1;
  int S0, S1;
  v4s mask_filter, mask_padding;

  mask_filter[0] = 0x03;
  mask_filter[1] = 0x00;
  mask_filter[2] = 0x01;
  mask_filter[3] = 0x02;

  mask_padding[0] = 0x01;
  mask_padding[1] = 0x02;
  mask_padding[2] = 0x00;
  mask_padding[3] = 0x03;

  uint8_t *In_Img_ptr0;
  uint8_t *In_Img_ptr1;
  uint8_t *In_Img_ptr2;

  v4s coef_v00, coef_v01, coef_v02;
  v4s coef_v00_l, coef_v01_l, coef_v02_l;
  v4s coef_v00_r, coef_v01_r, coef_v02_r;
  v4s coef_v10, coef_v11, coef_v12;

  for(ch=lb; ch < ub; ch++) {

    // use three pointers sweeping three lines of the image together
    In_Img_ptr0 = (In_Img + ch*R*C); // input layout is CHW
    In_Img_ptr1 = (In_Img + ch*R*C + C);
    In_Img_ptr2 = (In_Img + ch*R*C + 2*C);

    coef_v00 = *(v4s *) (Kernel + ch*3*3); // filter -> K0K1K2K3 
    coef_v01 = *(v4s *) (Kernel + 3 + ch*3*3); // filter layout is CHW
    coef_v02 = *(v4s *) (Kernel + 6 + ch*3*3);

    coef_v00[3] = 0; coef_v01[3] = 0; coef_v02[3] = 0; // filter 1 -> K0K1K2x

    coef_v10 = __builtin_pulp_shuffleb(coef_v00, mask_filter); // filter 2 -> xK0K1K2
    coef_v11 = __builtin_pulp_shuffleb(coef_v01, mask_filter);
    coef_v12 = __builtin_pulp_shuffleb(coef_v02, mask_filter);

    coef_v00_l = __builtin_pulp_shuffleb(coef_v00, mask_padding); // for left padding -> K1K2xx
    coef_v01_l = __builtin_pulp_shuffleb(coef_v01, mask_padding);
    coef_v02_l = __builtin_pulp_shuffleb(coef_v02, mask_padding);
    coef_v00_l[2] = 0; coef_v01_l[2] = 0; coef_v02_l[2] = 0;

    coef_v00_r = coef_v00; // for right padding -> K0K1xx
    coef_v01_r = coef_v01;
    coef_v02_r = coef_v02;
    coef_v00_r[2] = 0; coef_v01_r[2] = 0; coef_v02_r[2] = 0;

    for (r=0; r < (R - 2); r++) {  

      v4u data_v0, data_v1, data_v2;

      if (p_l) {

        S0 = 0;
        t0 = r*C*CH + ch; // ouput layout is HWC

        // for data_v0, LD + LBU is faster than packing
        data_v0 = *(v4u *) In_Img_ptr0;
        data_v1 = *(v4u *) In_Img_ptr1;
        data_v2 = *(v4u *) In_Img_ptr2;

        /* VECTORIZED LOOP */
        S0 = __builtin_pulp_sdotusp4(data_v0, coef_v00_l, S0);
        S0 = __builtin_pulp_sdotusp4(data_v1, coef_v01_l, S0);
        S0 = __builtin_pulp_sdotusp4(data_v2, coef_v02_l, S0);

        if (flag_batch_norm && flag_relu)
        {
          S0 = pulp_nn_bn_quant_u8(S0, *(kappa + ch), *(lambda + ch), out_shift);
        }
        else
        {
          if(flag_relu == 1)
          {
            S0 = pulp_nn_quant_u8(S0, out_mult, out_shift);
          }
          else
          {
            S0 = (uint8_t) clip8(S0 >> out_shift);
          }
        }

        Out_Img[t0] = (uint8_t)(S0);
      }

      for (c=p_l; c < C - p_r; c+=2) {

        S0 = 0;
        S1 = 0;
        t0 = r*C*CH + c*CH + ch;
        t1 = r*C*CH + (c + 1)*CH + ch;

        // for data_v0, LD + LBU is faster than packing
        data_v0 = *(v4u *) In_Img_ptr0;
        data_v1 = *(v4u *) In_Img_ptr1;
        data_v2 = *(v4u *) In_Img_ptr2;
        
        /* VECTORIZED LOOP */
        S0 = __builtin_pulp_sdotusp4(data_v0, coef_v00, S0);
        S0 = __builtin_pulp_sdotusp4(data_v1, coef_v01, S0);
        S0 = __builtin_pulp_sdotusp4(data_v2, coef_v02, S0);
        S1 = __builtin_pulp_sdotusp4(data_v0, coef_v10, S1);
        S1 = __builtin_pulp_sdotusp4(data_v1, coef_v11, S1);
        S1 = __builtin_pulp_sdotusp4(data_v2, coef_v12, S1);        

        if (flag_batch_norm && flag_relu)
        {
          S0 = pulp_nn_bn_quant_u8(S0, *(kappa + ch), *(lambda + ch), out_shift);
          S1 = pulp_nn_bn_quant_u8(S1, *(kappa + ch), *(lambda + ch), out_shift);
        }
        else
        {
          if(flag_relu == 1)
          {
            S0 = pulp_nn_quant_u8(S0, out_mult, out_shift);
            S1 = pulp_nn_quant_u8(S1, out_mult, out_shift);
          }
          else
          {
            S0 = (uint8_t) clip8(S0 >> out_shift);
            S1 = (uint8_t) clip8(S1 >> out_shift);
          }
        }

        Out_Img[t0] = (uint8_t)(S0);
        Out_Img[t1] = (uint8_t)(S1);

        In_Img_ptr0+=2; // next pixel on the right
        In_Img_ptr1+=2;
        In_Img_ptr2+=2;
      }
      if (p_r) {      

        S0 = 0;
        t0 = r*C*CH + c*CH + ch;

        // for data_v0, LD + LBU is faster than packing
        data_v0 = *(v4u *) In_Img_ptr0;
        data_v1 = *(v4u *) In_Img_ptr1;
        data_v2 = *(v4u *) In_Img_ptr2;

        /* VECTORIZED LOOP */
        S0 = __builtin_pulp_sdotusp4(data_v0, coef_v00_r, S0);
        S0 = __builtin_pulp_sdotusp4(data_v1, coef_v01_r, S0);
        S0 = __builtin_pulp_sdotusp4(data_v2, coef_v02_r, S0);

        if (flag_batch_norm && flag_relu)
        {
          S0 = pulp_nn_bn_quant_u8(S0, *(kappa + ch), *(lambda + ch), out_shift);
        }
        else
        {
          if(flag_relu == 1)
          {
            S0 = pulp_nn_quant_u8(S0, out_mult, out_shift);
          }
          else
          {
            S0 = (uint8_t) clip8(S0 >> out_shift);
          }
        }

        Out_Img[t0] = (uint8_t)(S0);
      }

      In_Img_ptr0+=2; // next pixel on the right
      In_Img_ptr1+=2;
      In_Img_ptr2+=2;    
    }  
  }
  pi_cl_team_barrier(0);
}
