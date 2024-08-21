/*
 * pulp_nn_utils.h
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

#ifdef GAP_SDK
#include "pulp.h"
#endif

uint8_t pulp_nn_bn_quant_u8 (int32_t phi, int64_t k, int64_t lambda, int8_t  d);

uint8_t pulp_nn_quant_u8(int32_t phi, int16_t m, int8_t  d);

uint8_t pulp_nn_add_quant_u8(uint8_t pix1,uint8_t pix2,int16_t m1,int16_t m2,int8_t  d);

void pulp_nn_im2col_int8_dmafree(uint8_t * pInput, uint8_t * pOutput, unsigned int blockSize);

void pulp_nn_im2col_int8(uint8_t * pInput, uint8_t * pOutput, unsigned int blockSize);

void pulp_nn_compare_and_replace_if_larger_int8(uint8_t * base,uint8_t * target,uint16_t length);

void pulp_nn_avg_and_replace_int8(int8_t * base,int8_t * target,uint16_t length);

uint8_t pulp_nn_bn_quant_u4 (int32_t phi, int64_t k, int64_t lambda, int8_t  d);

uint8_t pulp_nn_quant_u4(int32_t phi, int16_t m, int8_t  d);

uint8_t pulp_nn_bn_quant_u2 (int32_t phi, int64_t k, int64_t lambda, int8_t  d);

uint8_t pulp_nn_quant_u2(int32_t phi, int16_t m, int8_t  d);

v4s pulp_nn_i4_to_i8_r( int8_t *pSrc);

v4s pulp_nn_i2_to_i8_r( int8_t *pSrc);

v4u pulp_nn_u4_to_u8_r(uint8_t *pSrc);

v4u pulp_nn_u2_to_u8_r(uint8_t *pSrc);

void pulp_nn_i4_to_i8( int8_t *pSrc, int8_t *pDst);

void pulp_nn_i2_to_i8( int8_t * pSrc, int8_t * pDst);

void pulp_nn_u4_to_u8(uint8_t *pSrc, uint8_t *pDst);

void pulp_nn_u2_to_u8(uint8_t * pSrc, uint8_t * pDst);

void pulp_zero_mem(uint8_t * pBuffer, unsigned int size);

void pulp_nn_im2col_u8_to_u8(uint8_t * pInput, uint8_t * pOutput, unsigned int blockSize);

void pulp_nn_im2col_u4_to_u8(uint8_t * pInput, uint8_t * pOutput, unsigned int blockSize);

void pulp_nn_im2col_u2_to_u8(uint8_t * pInput, uint8_t * pOutput, unsigned int blockSize);

int8_t pulp_nn_i4_quant(int input, int16_t * pThr);

int8_t pulp_nn_i2_quant(int input, int16_t * pThr);
