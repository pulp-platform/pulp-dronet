/*
 * dory.h
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

#include "mchan_test.h"
#include "pulp.h"
unsigned int dory_get_tile_1d(
  unsigned x,
  int tile_ii,
  int tile_size_i,
  int data_size
);
unsigned int dory_get_tile_2d(
  unsigned int x,
  int tile_ii,
  int tile_jj,
  int tile_size_i,
  int tile_size_j,
  int tile_stride_j,
  int data_size
);
unsigned int dory_get_tile_3d(
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
);

unsigned int dory_get_tile_4d(
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
);

void dory_dma_memcpy_3d_custom(
  unsigned int ext,
  unsigned int loc,
  unsigned short size,
  unsigned short stride_1,
  unsigned short stride_0,
  unsigned short length_2,
  unsigned short length_0,
  unsigned int dir,
  unsigned int *id
);

void dory_dma_memcpy_3d_custom_hwc_to_chw(
  unsigned int ext,
  unsigned int loc,
  unsigned short size,
  unsigned short stride_1,
  unsigned short stride_0,
  unsigned short length_2,
  unsigned short length_0,
  unsigned int dir,
  unsigned int *id
);