/*
 * layer_template.c
 * Alessio Burrello <alessio.burrello@unibo.it>
 * Francesco Conti <f.conti@unibo.it>
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

#include "layerMatMul14_last.h"
#define VERBOSE_PRINT(...) printf(__VA_ARGS__)

void layerMatMul14_last(
  void *args
) {
  //////////////////////////////////////////////////////////////////////////
  // arguments assigning: keeping same interface between L2 and L3 memory //
  //////////////////////////////////////////////////////////////////////////
  unsigned int *real_arg = (unsigned int *) args;
  unsigned int l3_x =(unsigned int)  real_arg[0];
  unsigned int l3_y =(unsigned int)  real_arg[1];
  unsigned int l3_W =(unsigned int)  real_arg[2];
  unsigned int l2_x =(unsigned int)  real_arg[3];
  unsigned int l2_x_2 =(unsigned int)  real_arg[4];
  unsigned int l2_y =(unsigned int)  real_arg[5];
  unsigned int l2_W =(unsigned int)  real_arg[6];
  unsigned int l1_buffer =(unsigned int)  real_arg[7];
  unsigned int hyperram =(unsigned int)  real_arg[8];
  unsigned int out_mult_in =(unsigned int)  real_arg[9];
  unsigned int inmul1 = (unsigned int) real_arg[10];
  unsigned int inmul2 = (unsigned int) real_arg[11];
  unsigned int out_shift_in = (unsigned int) real_arg[12];

  //////////////////////////
  // Variable declaration //
  //////////////////////////
  unsigned int dma_evt;
  volatile int p_r, p_l, p_t, p_b;
  volatile unsigned short  W_tile_size_nof;
  volatile unsigned short  W_tile_size_nif;
  volatile unsigned short  W_tile_size_byte;
  volatile unsigned short W_length_nif_byte;
  volatile char *x;
  volatile char *W;
  volatile char *y;
  volatile char *b;
  volatile int x_tile_size_nif_exec;
  volatile int x_tile_size_h_exec;
  volatile int x_tile_size_w_exec;
  volatile int y_tile_size_nof;
  volatile int y_tile_size_h;
  volatile int y_tile_size_w;
  volatile int y_tile_size_byte;
  volatile int y_length_nof_byte;
  volatile int db_x;
  volatile int db_W;
  volatile int db_act;
  volatile int db_y;
  volatile int exec_db_x;
  volatile int exec_db_W;
  volatile int exec_db_act;
  volatile pi_cl_dma_copy_t copy_k;
  volatile pi_cl_dma_copy_t copy_lambda;
  // double buffering state
  int db_state_x=0;
  int db_state_W=0;
  int db_state_y=1;
  // last-tile flags
  int iter;
  // tile loop indeces
  int _i_nof_load=0, _i_nif_load=0, _i_h_load=0, _i_w_load=0;
  int _i_nof_exec=0, _i_nif_exec=0, _i_h_exec=0, _i_w_exec=0;
  volatile char *im2col;
  im2col = l1_buffer + 2400;
  /////////////////////////////////////
  /// Not Double buffered transfers ///
  /////////////////////////////////////
  pi_cl_team_barrier(0);
  //////////////////////////////////////////////////////////////
  // Allocation of one channel per each core for DMA transfer //
  //////////////////////////////////////////////////////////////
  dma_evt = mchan_alloc();
  ////////////////////////////
  // First tile transfering //
  ////////////////////////////
  dory_dma_memcpy_3d_custom(
  l2_x, // ext
  (l1_buffer + 0) + 0, // loc
  784, // size: dimension of the buffer
  784, // stride_1: stride for the 3d copy: if we have to copy on n_features axis, this is the stride to change from first 2D space to the next ones.
  784, // stride_0: stride to be passed to 2d_copy: the dimension w of the in image
  1,// length_2: how many 2_d copies we need -> the dimension of the tile in n_features direction
  784, // length_0: legnth of the 1_d copy, the length of tile in w direction
  1, // dir
  &dma_evt // copy
  );
  dory_dma_memcpy_3d_custom(
  l2_W, // ext
  (l1_buffer + 800) + 0, // loc offset caused by size of tile_x*2 (double_buffer) and tile_y*2 (double buffer)
  1568, // size: dimension of matrix of weight * bytes_per_weight
  784, // stride_1: stride for the 3d copy: if we have to copy on n_features axis, this is the stride to change from first 2D space to the next ones.
  784, // stride_0: stride to be passed to 2d_copy: the dimension w of the in image
  2, // length_2: how many 2_d copies we need -> the dimension of the tile in n_features direction
  784, // length_0: legnth of the 1_d copy, the length of tile in w direction
  1, // dir
  &dma_evt // copy
  );
  mchan_barrier(dma_evt);
  pi_cl_team_barrier(0);


  // tile loop nest
  for(iter=0; iter<1*1*1*1; iter++) {
    // loop nest is nof,h,w,(nif=0)
    _i_w_load += 1;
    if(_i_w_load==1) 
    {
      _i_w_load = 0;
      _i_h_load += 1;
      if(_i_h_load==1) 
      {
        _i_h_load = 0;
        _i_nof_load += 1;
      }
    }
    // check if last in any dimension

    // compute double buffering offsets and update db state
    db_x = !db_state_x ? 784 : 0;
    db_W = !db_state_W ? 1568 : 0;
    db_y = !db_state_y ? 8 : 0;
    exec_db_x = 0;
    db_state_x = ! db_state_x;
    exec_db_W = db_state_W ? 1568 : 0;
    if (_i_nif_load!=_i_nif_exec || _i_nof_load!=_i_nof_exec)
      db_state_W = ! db_state_W;
    //switch all double buffering offset and y only after that all n_input_features have been analyzed: we need to pass all n_in to produce a single fil
///////// POSSIBLE BUG FIX!!!!! DB_STATE_Y NOT SWITCHED /////////////

    // double buffered reads
    if(iter<1*1*1*1-1) 
    {
      y_tile_size_h   = (_i_h_load+1 == 1)   ? 1 : 1;
      y_tile_size_w   = (_i_w_load+1 == 1)   ? 1 : 1;
      W_tile_size_nof = (_i_nof_load+1 == 1) ? 2 : 2;
      W_tile_size_nif = (_i_nif_load+1 == 1) ? 784 : 784;
      W_tile_size_byte = W_tile_size_nof*W_tile_size_nif*8*1*1/8;
      W_length_nif_byte = (_i_nif_load+1 == 1) ? 784 : 784;
    // transfer of next input tile in double buffering
      // transfer of next weight tile if changed input or output channels
      if (_i_nif_load!=_i_nif_exec || _i_nof_load!=_i_nof_exec)
      {
        dory_dma_memcpy_3d_custom_weights(
        dory_get_tile_3d(l2_W, _i_nof_load, 0, _i_nif_load, 2, 1*1, 784, 1*1, 784, 0,0,0,0,0,0, 8), // ext
        (l1_buffer + 800) + db_W, // loc
        W_tile_size_byte, // size: dimension of matrix of weight * bytes_per_weight
        784, // stride_1: stride for the 3d copy: if we have to copy on n_features axis, this is the stride to change from first 2D space to the next ones.
        784, // stride_0: stride to be passed to 2d_copy: the dimension w of the in image
        W_tile_size_nof, // length_2: how many 2_d copies we need -> the dimension of the tile in n_features direction
        W_length_nif_byte, // length_0: legnth of the 1_d copy, the length of tile in w direction
        1, // dir
        &dma_evt // copy
        );
      }
    }
    // creation of the pointers to input, output, weights, lambda and k
    x = (char *) (l1_buffer + 0 + exec_db_x);
    W = (char *) (l1_buffer + 800 + exec_db_W);
    y = (char *) (l1_buffer + 788 + db_y);
    // parameter passed to the kernel. Input and output sizes
    x_tile_size_nif_exec = (_i_nif_exec+1 == 1) ? 784 : 784;
    x_tile_size_h_exec   = (_i_h_exec+1 == 1)   ? 1 : 1;
    x_tile_size_w_exec   = (_i_w_exec+1 == 1)   ? 1 : 1;
    y_tile_size_nof = (_i_nof_exec+1 == 1) ? 2 : 2;
    y_tile_size_h   = (_i_h_exec+1 == 1)   ? 1 : 1;
    y_tile_size_w   = (_i_w_exec+1 == 1)   ? 1 : 1;
    y_tile_size_byte = y_tile_size_nof*y_tile_size_h*y_tile_size_w*32/8;
    y_length_nof_byte = (_i_nof_exec+1 == 1)   ? 8 : 8;
    p_r = 0;
    p_l = 0;
    p_t = 0;
    p_b = 0;
    if (_i_h_exec == 0)
      p_t = 0;
    if (_i_w_exec == 0)
      p_l = 0;
    if (_i_h_exec == 1-1)
      p_b = 0;
    if (_i_w_exec == 1-1)
      p_r = 0;

    pi_cl_team_barrier(0);
    asm volatile("": : :"memory");
    pulp_nn_linear_out_32( 
    x,
    W,
    x_tile_size_nif_exec,
    y_tile_size_nof,
    0, 0, 1, 1, 0, 0,
    y,
    0, 0, &dma_evt );
    pi_cl_team_barrier(0);
      // wait for DMA write/read
      mchan_barrier(dma_evt);

        dory_dma_memcpy_3d_custom_out(
        dory_get_tile_3d(l2_y, _i_h_exec, _i_w_exec, _i_nof_exec, 1, 1, 2, 1, 2, 0, 0, 0, 0, 0, 0, 32), // ext
        (l1_buffer + 788) + db_y, // loc
        y_tile_size_byte, // size
        8, // stride_1
        8, // stride_0
        y_tile_size_h, // length_2
        y_length_nof_byte, // length_0
        0, // dir
        &dma_evt // copy
        );
    // update prev iterators
    db_state_y = ! db_state_y; 
    _i_nof_exec = _i_nof_load;
    _i_nif_exec = _i_nif_load;
    _i_h_exec = _i_h_load;
    _i_w_exec = _i_w_load;
    pi_cl_team_barrier(0);
  }

  // wait for final write
  mchan_barrier(dma_evt);
  mchan_free(dma_evt);
}
