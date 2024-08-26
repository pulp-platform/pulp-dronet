/*
 * pulp_nn_utils.c
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
#ifdef PULPNN_USE_DMA
#include "mchan_test.h"
#endif

#define bitext(x,size,off) __builtin_pulp_bextract(x,size,off)
#define bitextu(x,size,off) __builtin_pulp_bextractu(x,size,off)
#define pack(x,y,z,t)      __builtin_pulp_pack4(x,y,z,t)
#define max4(a,b)  		    __builtin_pulp_maxu4(a,b)
#define avg4(a,b)         __builtin_pulp_avg4(a,b)
#define clip8(x) __builtin_pulp_clipu_r(x, 255)

uint8_t __attribute__((always_inline)) pulp_nn_bn_quant_u8 (
  int32_t phi,
  int64_t k,
  int64_t lambda,
  int8_t  d
) {
  int64_t integer_image_phi = (k * phi) + lambda;
  int64_t x = (integer_image_phi) >> d;
  uint8_t res = __builtin_pulp_clipu_r(x,255);
  return res;
}

uint8_t pulp_nn_add_quant_u8 (
  uint8_t pix1,
  uint8_t pix2,
  int16_t m1,
  int16_t m2,
  int8_t  d
) {
  /* Integer Batch Normalization */
  uint32_t integer_image = pix1*m1 + pix2*m2;
  /* Quantization */
  uint16_t x = (integer_image) >> d;
  uint8_t res = clip8(x);
  return res;
}

void pulp_nn_compare_and_replace_if_larger_int8(uint8_t * base,
						                                    uint8_t * target,
						                                    uint16_t length)
{
  uint8_t *pIn = base;
  uint8_t *pCom = target;
  v4u inp;
  v4u com;
  uint16_t cnt = length >> 2;

  while(cnt > 0u)
  {
    inp = *((v4u*)pIn);
    com = *((v4u*)pCom);
    pCom+=4;

    *((v4u*)pIn) = max4(inp, com);
    pIn+=4;
    cnt--;
  }

  uint16_t left = length & 0x3;
  while (left>0u)
  {
    if(*pIn<*pCom)
      *pIn=*pCom;
    pIn++;
    pCom++;
    left--;
  }
}


void pulp_nn_avg_and_replace_int8(int8_t * base,
                                  int8_t * target,
                                  uint16_t length)
{
  int8_t *pIn = base;
  int8_t *pCom = target;
  v4s inp;
  v4s com;
  uint16_t cnt = length >> 2;

  while(cnt > 0u)
  {
    inp = *((v4s*)pIn);
    com = *((v4s*)pCom);
    pCom+=4;

    *((v4s*)pIn) = avg4(inp, com);
    pIn+=4;
    cnt--;
  }
}


#ifdef PULPNN_USE_DMA
void pulp_nn_im2col_int8(uint8_t * pInput, uint8_t * pOutput, unsigned int blockSize)
{
#if (MCHAN_VERSION < 7)
  mchan_transfer(blockSize, 1, 1, 0, 1, 0, 0, (unsigned int) pInput, (unsigned int) pOutput, 0, 0);
#elif (MCHAN_VERSION == 7)
  mchan_transfer(blockSize, 1, 1, 0, 0, 1, 0, 0, (unsigned int) pInput, (unsigned int) pOutput, 0, 0, 0, 0);
#endif
}
#else
void pulp_nn_im2col_int8(uint8_t * pInput, uint8_t * pOutput, unsigned int blockSize)
{
  unsigned int blkCnt = blockSize >> 2u;
  unsigned int lfover = blockSize & 0x3;

  for (unsigned int i = 0; i<blkCnt; i++)
  {
    *((v4s*)pOutput) = *((v4s*) pInput);
    pInput+=4;
    pOutput+=4;
  }
  while(lfover)
  {
    *((uint8_t*)pOutput) = *((uint8_t*)pInput);
    pOutput++;
    pInput++;
    lfover--;
  }
}
#endif

uint8_t __attribute__((always_inline)) pulp_nn_quant_u8(
  int32_t phi,
  int16_t m,
  int8_t  d
) {
  int16_t x = (m * phi) >> d;
  uint8_t res = __builtin_pulp_clipu_r(x,255);
  return res;
}

uint8_t __attribute__((always_inline)) pulp_nn_bn_quant_u4 (
  int32_t phi,
  int64_t k,
  int64_t lambda,
  int8_t  d
) {
  int64_t integer_image_phi = (k * phi) + lambda;
  int64_t x = (integer_image_phi) >> d;
  uint8_t res = __builtin_pulp_clipu_r(x,15);
  return res;
}

uint8_t __attribute__((always_inline)) pulp_nn_quant_u4(
  int32_t phi,
  int16_t m,
  int8_t  d
) {
  int16_t x = (m * phi) >> d;
  uint8_t res = __builtin_pulp_clipu_r(x,15);
  return res;
}

uint8_t __attribute__((always_inline)) pulp_nn_bn_quant_u2 (
  int32_t phi,
  int64_t k,
  int64_t lambda,
  int8_t  d
) {
  int64_t integer_image_phi = (k * phi) + lambda;
  int64_t x = (integer_image_phi) >> d;
  uint8_t res = __builtin_pulp_clipu_r(x,3);
  return res;
}

uint8_t __attribute__((always_inline)) pulp_nn_quant_u2(
  int32_t phi,
  int16_t m,
  int8_t  d
) {
  int16_t x = (m * phi) >> d;
  uint8_t res = __builtin_pulp_clipu_r(x,3);
  return res;
}


v4s __attribute__((always_inline))pulp_nn_i4_to_i8_r( int8_t *pSrc)
{
	v4s Src = *((v4s*) pSrc);
	int8_t bext1, bext2, bext3, bext4;

	bext1 = (int8_t) bitext((int) Src, 4, 0);
	bext2 = (int8_t) bitext((int) Src, 4, 4);
	bext3 = (int8_t) bitext((int) Src, 4, 8);
	bext4 = (int8_t) bitext((int) Src, 4, 12);
	v4s res = pack((int8_t) bext1, (int8_t) bext2, (int8_t) bext3, (int8_t) bext4);

	return res;
}

v4u __attribute__((always_inline))pulp_nn_u4_to_u8_r(uint8_t *pSrc)
{
	v4u Src = *((v4u*) pSrc);
	uint8_t bext1, bext2, bext3, bext4;

	bext1 = (uint8_t) bitextu((unsigned int) Src, 4, 0);
	bext2 = (uint8_t) bitextu((unsigned int) Src, 4, 4);
	bext3 = (uint8_t) bitextu((unsigned int) Src, 4, 8);
	bext4 = (uint8_t) bitextu((unsigned int) Src, 4, 12);
	v4u res = pack((uint8_t) bext1, (uint8_t) bext2, (uint8_t) bext3, (uint8_t) bext4);

	return res;
}

v4s __attribute__((always_inline))pulp_nn_i2_to_i8_r( int8_t *pSrc)
{
	v4s Src = *((v4s*) pSrc);
	int8_t bext1, bext2, bext3, bext4;

	bext1 = (int8_t) bitext((int) Src, 2, 0);
	bext2 = (int8_t) bitext((int) Src, 2, 2);
	bext3 = (int8_t) bitext((int) Src, 2, 4);
	bext4 = (int8_t) bitext((int) Src, 2, 6);
	v4s res = pack((int8_t) bext1, (int8_t) bext2, (int8_t) bext3, (int8_t) bext4);

	return res;
}

v4u __attribute__((always_inline))pulp_nn_u2_to_u8_r(uint8_t *pSrc)
{
	v4u Src = *((v4u*) pSrc);
	uint8_t bext1, bext2, bext3, bext4;

	bext1 = (uint8_t) bitextu((unsigned int) Src, 2, 0);
	bext2 = (uint8_t) bitextu((unsigned int) Src, 2, 2);
	bext3 = (uint8_t) bitextu((unsigned int) Src, 2, 4);
	bext4 = (uint8_t) bitextu((unsigned int) Src, 2, 6);
	v4u res = pack((uint8_t) bext1, (uint8_t) bext2, (uint8_t) bext3, (uint8_t) bext4);

	return res;
}

void __attribute__((always_inline))pulp_nn_i4_to_i8( int8_t *pSrc, int8_t *pDst)
{
	v4s Src = *((v4s*) pSrc);
	int8_t bext1, bext2, bext3, bext4;
	bext1 = (int8_t) bitext((int) Src, 4, 0);
	bext2 = (int8_t) bitext((int) Src, 4, 4);
	bext3 = (int8_t) bitext((int) Src, 4, 8);
	bext4 = (int8_t) bitext((int) Src, 4, 12);
	*((v4s*)pDst) = pack((int8_t) bext1, (int8_t) bext2, (int8_t) bext3, (int8_t) bext4);
  	asm volatile(""::: "memory");
	pDst+=4;
	bext1 = (int8_t) bitext((int) Src, 4, 16);
	bext2 = (int8_t) bitext((int) Src, 4, 20);
	bext3 = (int8_t) bitext((int) Src, 4, 24);
	bext4 = (int8_t) bitext((int) Src, 4, 28);
	*((v4s*)pDst) = pack((int8_t) bext1, (int8_t) bext2, (int8_t) bext3, (int8_t) bext4);
}

void __attribute__((always_inline))pulp_nn_u4_to_u8(uint8_t *pSrc, uint8_t *pDst)
{
	v4u Src = *((v4u*) pSrc);
	uint8_t bext1, bext2, bext3, bext4;
	bext1 = (uint8_t) bitextu((unsigned int) Src, 4, 0);
	bext2 = (uint8_t) bitextu((unsigned int) Src, 4, 4);
	bext3 = (uint8_t) bitextu((unsigned int) Src, 4, 8);
	bext4 = (uint8_t) bitextu((unsigned int) Src, 4, 12);
	*((v4u*)pDst) = pack((uint8_t) bext1, (uint8_t) bext2, (uint8_t) bext3, (uint8_t) bext4);
  	asm volatile(""::: "memory");
	pDst+=4;
	bext1 = (uint8_t) bitextu((unsigned int) Src, 4, 16);
	bext2 = (uint8_t) bitextu((unsigned int) Src, 4, 20);
	bext3 = (uint8_t) bitextu((unsigned int) Src, 4, 24);
	bext4 = (uint8_t) bitextu((unsigned int) Src, 4, 28);
	*((v4u*)pDst) = pack((uint8_t) bext1, (uint8_t) bext2, (uint8_t) bext3, (uint8_t) bext4);
}

void __attribute__((always_inline))pulp_nn_i2_to_i8( int8_t * pSrc, int8_t * pDst)
{
	v4s Src = *((v4s*) pSrc);
	int8_t bext1, bext2, bext3, bext4;
	bext1 = (int8_t) bitext((int) Src, 2, 0);
	bext2 = (int8_t) bitext((int) Src, 2, 2);
	bext3 = (int8_t) bitext((int) Src, 2, 4);
	bext4 = (int8_t) bitext((int) Src, 2, 6);
	*((v4s*)pDst) = pack((int8_t) bext1, (int8_t) bext2, (int8_t) bext3, (int8_t) bext4);
  	asm volatile(""::: "memory");
	pDst+=4;
	bext1 = (int8_t) bitext((int) Src, 2, 8);
	bext2 = (int8_t) bitext((int) Src, 2, 10);
	bext3 = (int8_t) bitext((int) Src, 2, 12);
	bext4 = (int8_t) bitext((int) Src, 2, 14);
	*((v4s*)pDst) = pack((int8_t) bext1, (int8_t) bext2, (int8_t) bext3, (int8_t) bext4);
  	asm volatile(""::: "memory");
	pDst+=4;
	bext1 = (int8_t) bitext((int) Src, 2, 16);
	bext2 = (int8_t) bitext((int) Src, 2, 18);
	bext3 = (int8_t) bitext((int) Src, 2, 20);
	bext4 = (int8_t) bitext((int) Src, 2, 22);
	*((v4s*)pDst) = pack((int8_t) bext1, (int8_t) bext2, (int8_t) bext3, (int8_t) bext4);
  	asm volatile(""::: "memory");
	pDst+=4;
	bext1 = (int8_t) bitext((int) Src, 2, 24);
	bext2 = (int8_t) bitext((int) Src, 2, 26);
	bext3 = (int8_t) bitext((int) Src, 2, 28);
	bext4 = (int8_t) bitext((int) Src, 2, 30);
	*((v4s*)pDst) = pack((int8_t) bext1, (int8_t) bext2, (int8_t) bext3, (int8_t) bext4);
}

void __attribute__((always_inline))pulp_nn_u2_to_u8(uint8_t * pSrc, uint8_t * pDst)
{
	v4u Src = *((v4u*) pSrc);
	uint8_t bext1, bext2, bext3, bext4;
	bext1 = (uint8_t) bitextu((unsigned int) Src, 2, 0);
	bext2 = (uint8_t) bitextu((unsigned int) Src, 2, 2);
	bext3 = (uint8_t) bitextu((unsigned int) Src, 2, 4);
	bext4 = (uint8_t) bitextu((unsigned int) Src, 2, 6);
	*((v4u*)pDst) = pack((uint8_t) bext1, (uint8_t) bext2, (uint8_t) bext3, (uint8_t) bext4);
  	asm volatile(""::: "memory");
	pDst+=4;
	bext1 = (uint8_t) bitextu((unsigned int) Src, 2, 8);
	bext2 = (uint8_t) bitextu((unsigned int) Src, 2, 10);
	bext3 = (uint8_t) bitextu((unsigned int) Src, 2, 12);
	bext4 = (uint8_t) bitextu((unsigned int) Src, 2, 14);
	*((v4u*)pDst) = pack((uint8_t) bext1, (uint8_t) bext2, (uint8_t) bext3, (uint8_t) bext4);
  	asm volatile(""::: "memory");
	pDst+=4;
	bext1 = (uint8_t) bitextu((unsigned int) Src, 2, 16);
	bext2 = (uint8_t) bitextu((unsigned int) Src, 2, 18);
	bext3 = (uint8_t) bitextu((unsigned int) Src, 2, 20);
	bext4 = (uint8_t) bitextu((unsigned int) Src, 2, 22);
	*((v4u*)pDst) = pack((uint8_t) bext1, (uint8_t) bext2, (uint8_t) bext3, (uint8_t) bext4);
  	asm volatile(""::: "memory");
	pDst+=4;
	bext1 = (uint8_t) bitextu((unsigned int) Src, 2, 24);
	bext2 = (uint8_t) bitextu((unsigned int) Src, 2, 26);
	bext3 = (uint8_t) bitextu((unsigned int) Src, 2, 28);
	bext4 = (uint8_t) bitextu((unsigned int) Src, 2, 30);
	*((v4u*)pDst) = pack((uint8_t) bext1, (uint8_t) bext2, (uint8_t) bext3, (uint8_t) bext4);
}

void __attribute__((always_inline))pulp_zero_mem(uint8_t * pBuffer, unsigned int size)
{
  int lfover = size &0x3;
  for (int i=0; i<(size>>2); i++)
  {
    *((v4u *)pBuffer) = (v4u){0,0,0,0};
    asm volatile("":::"memory");
    pBuffer+=4;
  }
  while(lfover)
  {
    *pBuffer++=0;
    lfover--;
  }
}

void __attribute__((always_inline))pulp_nn_im2col_u8_to_u8(uint8_t * pInput, uint8_t * pOutput, unsigned int blockSize)
{
  unsigned int blkCnt = blockSize >> 2u;
  unsigned int lfover = blockSize & 0x3;

  for (unsigned int i = 0; i<blkCnt; i++)
  {
    *((v4u*)pOutput) = *((v4u*) pInput);
    pInput+=4;
    pOutput+=4;
  }
  while(lfover)
  {
    *((uint8_t*)pOutput) = *((uint8_t*)pInput);
    pOutput++;
    pInput++;
    lfover--;
  }
}

void pulp_nn_im2col_u4_to_u8(uint8_t * pInput, uint8_t * pOutput, unsigned int blockSize)
{
  unsigned int blkCnt = blockSize >> 3u;
  unsigned int lfover = blockSize & 0x7;

  for (int i = 0; i<blkCnt; i++)
  {
    pulp_nn_u4_to_u8(pInput, pOutput);
    asm volatile("":::"memory");
    pInput+=4;
    pOutput+=8;
  }
  while(lfover)
  {
	*((uint8_t*)pOutput) = (uint8_t) bitextu((unsigned int) *pInput, 4, 0);
	pOutput++;
	*((uint8_t*)pOutput) = (uint8_t) bitextu((unsigned int) *pInput, 4, 4);
	pOutput++;
	pInput++;
	lfover-=2;
  }
}

void __attribute__((always_inline))pulp_nn_im2col_u2_to_u8(uint8_t * pInput, uint8_t * pOutput, unsigned int blockSize)
{
  unsigned int blkCnt = blockSize >> 4u;
  unsigned int lfover = blockSize & 0xf;

  for(int i = 0; i<blkCnt; i++)
  {
    pulp_nn_u2_to_u8(pInput, pOutput);
    pInput+=4;
    pOutput+=16;
  }
  while(lfover)
  {
	*((v4u*)pOutput) = pulp_nn_u2_to_u8_r(pInput);
	pInput++;
	pOutput+=4;
	lfover-=4;
  }
}

int8_t __attribute__ ((always_inline)) pulp_nn_i4_quant(int input, int16_t * pThr)
{
	if(input <= pThr[7] )
	{
		if( input <= pThr[3])
		{
			if( input <= pThr[1])
			{
				if( input <= pThr[0])
					return -8;
				else
					return -7;
			}
			else
			{
				if( input <= pThr[2])
					return -6;
				else
					return -5;
			}
		}
		else
		{
			if( input <= pThr[5])
			{
				if( input <= pThr[4])
					return -4;
				else
					return -3;
			}
			else
			{
				if( input <= pThr[6])
					return -2;
				else
					return -1;
			}
		}
	}
	else
	{
		if( input <= pThr[11])
		{
			if( input <= pThr[9])
			{
				if( input <= pThr[8])
					return 0;
				else
					return 1;
			}
			else
			{
				if( input <= pThr[10])
					return 2;
				else
					return 3;
			}
		}
		else
		{
			if( input <= pThr[13])
			{
				if( input <= pThr[12])
					return 4;
				else
					return 5;
			}
			else
			{
				if( input <= pThr[14])
					return 6;
				else
					return 7;
			}
		}
	}
}

int8_t __attribute__ ((always_inline)) pulp_nn_i2_quant(int input, int16_t * pThr)
{
	if( input <= pThr[1])
  {
		if( input <= pThr[0])
        {
			return -2;
		}
        else
        {
			return -1;
		}
	}
    else
    {
		if( input <= pThr[2])
        {
			return 0;
		}
        else
        {
			return 1;
		}
	}
}
