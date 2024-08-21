/*
 * dory.c
 * Alessio Burrello <alessio.burrello@unibo.it>
 *
 * Copyright (C) 2019-2020 University of Bologna
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

#include "dory.h"

/**
 *  @brief Gets a tile over a one-dimensional tiling grid.
 *
 *  Computes a pointer to the base of a particular tile in a one-dimensional
 *  tiling grid indexed by a (ii) index; in pseudo-Python
 *      ccn_get_tile_1d(x,ii) = x[ii*si:(ii+1)*si-1]
 *  where (si) os defining the pitch of the tiling grid in the (i) dimension.
 *
 *  @param x
 *      a pointer to the base of the 2d tiling grid.
 *  @param tile_ii
 *      the tiling index.
 *  @param tile_size_i
 *      the pitch of the tiling grid in the outer dimension, i.e. the distance
 *      between two "ticks" in the i dimension.
 *  @param data_size
 *      size of data in bytes
 */
unsigned int dory_get_tile_1d(
  unsigned x,
  int tile_ii,
  int tile_size_i,
  int data_size
) {
  unsigned int y = x + tile_ii*tile_size_i * data_size;
  return y;
}

/**
 *  @brief Gets a tile over a two-dimensional tiling grid.
 *
 *  Computes a pointer to the base of a particular tile in a two-dimensional
 *  tiling grid indexed by a (ii,jj) couple of indeces; in pseudo-Python
 *      ccn_get_tile_2d(x,ii,jj) = x[ii*si:(ii+1)*si-1,jj*sj:(jj+1)*sj-1]
 *  where (si,sj) is the couple defining the pitch of the tiling grid in the
 *  (i,j) dimensions.
 *
 *  @param *x
 *      a pointer to the base of the 2d tiling grid.
 *  @param tile_ii
 *      the tiling index in the outer dimension.
 *  @param tile_jj
 *      the tiling index in the inner dimension.
 *  @param tile_size_i
 *      the pitch of the tiling grid in the outer dimension, i.e. the distance
 *      between two "ticks" in the i dimension.
 *  @param tile_size_j
 *      the pitch of the tiling grid in the inner dimension, i.e. the distance
 *      between two "ticks" in the j dimension.
 *  @param tile_stride_j
 *      the total size of the tiling grid in the inner dimension, i.e. the 
 *      number of ticks in the j dimension.
 *  @param data_size
 *      size of data in bytes
 */
unsigned int  dory_get_tile_2d(
  unsigned int x,
  int tile_ii,
  int tile_jj,
  int tile_size_i,
  int tile_size_j,
  int tile_stride_j,
  int data_size
) {
  unsigned int y = x + tile_ii*tile_size_i * tile_stride_j * data_size
                     + tile_jj*tile_size_j * data_size;
  return y;
}

/**
 *  @brief Gets a tile over a three-dimensional tiling grid.
 *
 *  Computes a pointer to the base of a particular tile in a three-dimensional
 *  tiling grid indexed by a (ii,jj,kk) triple of indeces; in pseudo-Python
 *      ccn_get_tile_3d(x,ii,jj,kk) =
 *        x[ii*si:(ii+1)*si-1, jj*sj:(jj+1)*sj-1, kk*sk:(kk+1)*sk-1]
 *  where (si,sj,sk) is the triple defining the pitch of the tiling grid in the
 *  (i,j,k) dimensions.
 *
 *  @param *x
 *      a pointer to the base of the 2d tiling grid.
 *  @param tile_ii
 *      the tiling index in the outer dimension.
 *  @param tile_jj
 *      the tiling index in the middle dimension.
 *  @param tile_kk
 *      the tiling index in the inner dimension.
 *  @param tile_size_i
 *      the pitch of the tiling grid in the outer dimension, i.e. the distance
 *      between two "ticks" in the i dimension.
 *  @param tile_size_j
 *      the pitch of the tiling grid in the middle dimension, i.e. the distance
 *      between two "ticks" in the j dimension.
 *  @param tile_size_k
 *      the pitch of the tiling grid in the inner dimension, i.e. the distance
 *      between two "ticks" in the k dimension.
 *  @param tile_stride_j
 *      the total size of the tiling grid in the middle dimension, i.e. the 
 *      total number of ticks in the j dimension.
 *  @param tile_stride_k
 *      the total size of the tiling grid in the inner dimension, i.e. the 
 *      total number of ticks in the k dimension. 
 *  @param data_size
 *      size of data in bytes
 */
unsigned int  dory_get_tile_3d(
  unsigned int x,
  int tile_ii,
  int tile_jj,
  int tile_kk,
  int tile_size_i,
  int tile_size_j,
  int tile_size_k,
  int tile_stride_j,
  int tile_stride_k,
  int tile_overlap_i,
  int tile_overlap_j,
  int tile_overlap_k,
  int tile_offset_i,
  int tile_offset_j,
  int tile_offset_k,
  int data_size
) {
  unsigned int y = x + (tile_ii*(tile_size_i - tile_overlap_i) - tile_offset_i) * tile_stride_j * tile_stride_k * data_size / 8
                     + (tile_jj*(tile_size_j - tile_overlap_j) - tile_offset_j) * tile_stride_k * data_size / 8
                     + (tile_kk*(tile_size_k - tile_overlap_k) - tile_offset_k) * data_size / 8;
  return y;
}

unsigned int  dory_get_tile_4d(
  unsigned int x,
  int tile_ii,
  int tile_jj,
  int tile_kk,
  int tile_ll,
  int tile_size_i,
  int tile_size_j,
  int tile_size_k,
  int tile_size_l,
  int tile_stride_j,
  int tile_stride_k,
  int tile_stride_l,
  int tile_offset_i,
  int tile_offset_j,
  int tile_offset_k,
  int tile_offset_l,
  int data_size
) {
  unsigned int y = x + tile_ii*(tile_size_i - tile_offset_i) * tile_stride_j * tile_stride_k * tile_stride_l * data_size
                     + tile_jj*(tile_size_j - tile_offset_j) * tile_stride_k * tile_stride_l * data_size
                     + tile_kk*(tile_size_k - tile_offset_k) * tile_stride_l * data_size
                     + tile_ll*(tile_size_l - tile_offset_l) * data_size;
  return y;
}

#define MIN(a,b) ((a)<(b)?(a):(b))

void __attribute__ ((noinline)) dory_dma_memcpy_3d_custom_weights(
  unsigned int ext,
  unsigned int loc,
  unsigned short size,
  unsigned short stride_1,
  unsigned short stride_0,
  unsigned short length_2,
  unsigned short length_0,
  unsigned int dir,
  unsigned int *id
) 
{
  // parallelization
  if (pi_core_id()==0)
#if (MCHAN_VERSION < 7)
    mchan_transfer(size, dir, 1, 0, 1, 0, 0, (unsigned int)(ext), (unsigned int)(loc), 0, 0);
#elif (MCHAN_VERSION == 7)
    mchan_transfer(size, dir, 1, 0, 0, 1, 0, 0, (unsigned int)(ext), (unsigned int)(loc), 0, 0, 0, 0);
#endif
}

void __attribute__ ((noinline)) dory_dma_memcpy_3d_custom_out(
  unsigned int ext,
  unsigned int loc,
  unsigned short size,
  unsigned short stride_1,
  unsigned short stride_0,
  unsigned short length_2,
  unsigned short length_0,
  unsigned int dir,
  unsigned int *id
) 
{
  // parallelization
  int core_id = pi_core_id();
  int Log2Core = log2(NUM_CORES);
  int chunk = (length_2 >> Log2Core) + ((length_2 & (NUM_CORES-1))!=0);
  unsigned short length_1 = size / (length_2*length_0);
  int start_pixel, stop_pixel;
  start_pixel = MIN(chunk * core_id, length_2);
  stop_pixel = MIN(start_pixel+chunk, length_2);
  int offs_remote = stride_1*start_pixel;
  int offs_local = length_0*length_1*start_pixel;
  for ( int i=start_pixel; i<stop_pixel; i++) 
  {
    for ( int j=0; j<length_1; j++) 
    {
#if (MCHAN_VERSION < 7)
      mchan_transfer(length_0, dir, 1, 0, 1, 0, 0, (unsigned int)(ext + offs_remote), (unsigned int)(loc + offs_local), 0, 0);
#elif (MCHAN_VERSION == 7)
      mchan_transfer(length_0, dir, 1, 0, 0, 1, 0, 0, (unsigned int)(ext + offs_remote), (unsigned int)(loc + offs_local), 0, 0, 0, 0);
#endif
      offs_local  += length_0;
      offs_remote += stride_0;
    }
    offs_remote = offs_remote - stride_0*length_1 + stride_1;
  }
}

// copies are managed by 8 cores parallely. By now, 3d copies are a serie of 1d copy. In future, will be a serie of 2d copies.
void __attribute__ ((noinline)) dory_dma_memcpy_3d_custom(
  unsigned int ext,
  unsigned int loc,
  unsigned short size,
  unsigned short stride_1,
  unsigned short stride_0,
  unsigned short length_2,
  unsigned short length_0,
  unsigned int dir,
  unsigned int *id
) 
{
  // parallelization
  int core_id = pi_core_id();
  int Log2Core = log2(NUM_CORES);
  int chunk = (length_2 >> Log2Core) + ((length_2 & (NUM_CORES-1))!=0);
  unsigned short length_1 = size / (length_2*length_0);
  int start_pixel, stop_pixel;
  start_pixel = MIN(chunk * core_id, length_2);
  stop_pixel = MIN(start_pixel+chunk, length_2);
  int offs_remote = stride_1*start_pixel;
  int offs_local = length_0*length_1*start_pixel;
  for ( int i=start_pixel; i<stop_pixel; i++) 
  {
#if (MCHAN_VERSION < 7)
    mchan_transfer(length_0*length_1, dir, 1, 0, 1, 0, 0, (unsigned int)(ext + offs_remote), (unsigned int)(loc + offs_local), 0, 0);
#elif (MCHAN_VERSION == 7)
    mchan_transfer(length_0*length_1, dir, 1, 0, 0, 1, 0, 0, (unsigned int)(ext + offs_remote), (unsigned int)(loc + offs_local), 0, 0, 0, 0);
#endif
    offs_local  += length_0*length_1;
    offs_remote = offs_remote + stride_1;
  }
}

void __attribute__ ((noinline)) dory_dma_memcpy_3d_custom_blocking(
  unsigned int ext,
  unsigned int loc,
  unsigned short size,
  unsigned short stride_1,
  unsigned short stride_0,
  unsigned short length_2,
  unsigned short length_0,
  unsigned int dir,
  unsigned int *id
) 
{
  // parallelization
  int core_id = pi_core_id();
  int Log2Core = log2(NUM_CORES);
  int chunk = (length_2 >> Log2Core) + ((length_2 & (NUM_CORES-1))!=0);
  unsigned short length_1 = size / (length_2*length_0);
  int start_pixel, stop_pixel;
  start_pixel = MIN(chunk * core_id, length_2);
  stop_pixel = MIN(start_pixel+chunk, length_2);
  int offs_remote = stride_1*start_pixel;
  int offs_local = length_0*length_1*start_pixel;
  int dma_evt = mchan_alloc();
  for ( int i=start_pixel; i<stop_pixel; i++) 
  {
    for ( int j=0; j<length_1; j++) 
    {
      // alloc channels with barrier after if we consider v2 chips, with DMA issue 
#if (MCHAN_VERSION < 7)
      mchan_transfer(length_0, dir, 1, 0, 1, 0, 0, (unsigned int)(ext + offs_remote), (unsigned int)(loc + offs_local), 0, 0);
#elif (MCHAN_VERSION == 7)
      mchan_transfer(length_0, dir, 1, 0, 0, 1, 0, 0, (unsigned int)(ext + offs_remote), (unsigned int)(loc + offs_local), 0, 0, 0, 0);
#endif
      offs_local  += length_0;
      offs_remote += stride_0;
    }
    offs_remote = offs_remote - stride_0*length_1 + stride_1;
  }
  mchan_barrier(dma_evt);
  mchan_free(dma_evt);
}

// using DMA to move from a chw to an hwc layout. We use copies of 1 single bit at the time.
void __attribute__ ((noinline)) dory_dma_memcpy_3d_custom_hwc_to_chw(
  unsigned int ext,
  unsigned int loc,
  unsigned short size,
  unsigned short stride_1,
  unsigned short stride_0,
  unsigned short length_2,
  unsigned short length_0,
  unsigned int dir,
  unsigned int *id
) {
  int core_id = pi_core_id();
  int Log2Core = log2(NUM_CORES);
  int chunk = (length_0 >> Log2Core) + ((length_0 & (NUM_CORES-1))!=0);
  unsigned short length_1 = size / (length_2*length_0);
  int start_pixel, stop_pixel;
  start_pixel = MIN(chunk * core_id, length_0);
  stop_pixel = MIN(start_pixel+chunk, length_0);
  int offs_remote = start_pixel;
  int offs_local = length_2*length_1*start_pixel;
  int dma_evt = mchan_alloc();
  for ( int i=start_pixel; i<stop_pixel; i++) 
  {
#if (MCHAN_VERSION < 7)
    mchan_transfer(length_1*length_2, dir, 1, 1, 1, 0, 0, (unsigned int)(ext + offs_remote), (unsigned int)(loc + offs_local), 1, stride_0);
#elif (MCHAN_VERSION == 7)
    mchan_transfer(length_1*length_2, dir, 1, 1, 0, 1, 0, 0, (unsigned int)(ext + offs_remote), (unsigned int)(loc + offs_local), 1, stride_0, 0, 0);
#endif
    mchan_barrier(dma_evt);
    offs_local  += length_1*length_2;
    offs_remote = offs_remote + 1;
  }
  mchan_free(dma_evt);
}