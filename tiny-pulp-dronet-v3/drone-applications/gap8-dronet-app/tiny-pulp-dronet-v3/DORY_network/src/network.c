/*
 * network.c
 * Alessio Burrello <alessio.burrello@unibo.it>
 * Thorir Mar Ingolfsson <thoriri@iis.ee.ethz.ch>
 *
 * Modified for DRONET.
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
#include "mem_controller.h"
#include "network.h"
#include "pulp.h"
#include "dory.h"
#include "layerConvDWBNRelu10.h"
#include "layerConvDWBNRelu2.h"
#include "layerConvDWBNRelu6.h"
#include "layerMatMul14_last.h"
#include "layerConvBNRelu3.h"
#include "layerConvDWBNRelu12.h"
#include "layerConvBNRelu0.h"
#include "layerConvBNRelu9.h"
#include "layerMaxPool1.h"
#include "layerConvBNRelu11.h"
#include "layerConvBNRelu7.h"
#include "layerConvDWBNRelu8.h"
#include "layerConvDWBNRelu4.h"
#include "layerConvBNRelu13.h"
#include "pmsis.h"
#include "bsp/fs.h"
#include "bsp/fs/readfs.h"
#include "bsp/flash.h"
#include "bsp/ram.h"
#include "bsp/flash/hyperflash.h"
#include "bsp/ram/hyperram.h"

#define FLASH_BUFF_SIZE 128
// #define DEBUG_PRINT 1
// ADDED
extern int32_t *ResOut;


// allocation of buffers with parameters needed by the network execution
const char * L3_weights_files[] = {
  "ConvBNRelu0_weights.hex", "ConvDWBNRelu2_weights.hex", "ConvBNRelu3_weights.hex", "ConvDWBNRelu4_weights.hex", "ConvBNRelu5_weights.hex", "ConvDWBNRelu6_weights.hex", "ConvBNRelu7_weights.hex", "ConvDWBNRelu8_weights.hex", "ConvBNRelu9_weights.hex", "ConvDWBNRelu10_weights.hex", "ConvBNRelu11_weights.hex", "ConvDWBNRelu12_weights.hex", "ConvBNRelu13_weights.hex", "MatMul14_weights.hex"
};
int L3_weights_size[14];
static int L3_weights;
static int L3_input;
static int bypass_L3_input;
static int L3_output;
static int bypass_L3_output;
static int activations_input;
static int L3_layers[15] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
static int L3_input_layers[15] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
static int L3_output_layers[15] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
static int L3_weights_layers[15] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
static int allocate_layer[15] = {1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1};
static int branch_input[15] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
static int branch_output[15] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
static int branch_change[15] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
static int branch_last[15] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
static int check_weights[15] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
static int check_weights_dimension[15] = {164, 0, 100, 80, 100, 80, 100, 160, 200, 192, 200, 384, 400, 512, 1568};
static int cumulative_weights_dimension[15] = {0, 164, 164, 264, 344, 444, 524, 624, 784, 984, 1176, 1376, 1760, 2160, 2672};
static int check_activations[15] = {2556255, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
static int check_activations_dimension[15] = {80000, 40000, 10000, 2500, 2500, 2500, 2500, 676, 1352, 1352, 1352, 392, 784, 784, 784};  //dronet modification; increasing size of the first layer
static int check_activations_dimension_L3_in[15] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
static int check_activations_dimension_L3_out[15] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
static int out_mult_vector[15] = {1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0};
static int out_shift_vector[15] = {23, 0, 20, 21, 21, 22, 22, 21, 21, 22, 22, 22, 21, 23, 0};
static int inmul1_vector[15] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
static int inmul2_vector[15] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
static int check_activations_out[15] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
static int check_activations_out_dimension[15] = {40000, 10000, 2500, 2500, 2500, 2500, 676, 1352, 1352, 1352, 392, 784, 784, 784, 8};
static int layer_with_weights[15] = {1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1};

static uint8_t flashBuffer[FLASH_BUFF_SIZE];

static struct pi_hyperflash_conf flash_conf;
static struct pi_hyper_conf ram_conf;
static struct pi_device ram;




// filesystem management functions
void open_filesystem(struct pi_device *flash, struct pi_device *fs)
{
    struct pi_readfs_conf conf;
    struct pi_hyperflash_conf flash_conf;

    /* Init & open flash. */
    pi_hyperflash_conf_init(&flash_conf);
    pi_open_from_conf(flash, &flash_conf);
    if (pi_flash_open(flash))
    {
        printf("Error flash open !\n");
        pmsis_exit(-1);
    }

    /* Open filesystem on flash. */
    pi_readfs_conf_init(&conf);
    conf.fs.flash = flash;
    pi_open_from_conf(fs, &conf);
    if (pi_fs_mount(fs))
    {
        printf("Error FS mounting !\n");
        pmsis_exit(-2);
    }
}


int memId;
char* L2_output;
char* L2_input;
char* L2_weights_1;
char* L2_weights_2;
char* L2_buffer_allocation;
char* L2_buffer_tofree_copy;
int L2_buffer_allocation_end;
char *l1_buffer;
uint8_t * bypass_activations;
uint8_t * activation_to_keep;
char *exec_weights, *transfer_weights, *bypass_weights;
int L3_weights_internal;
//dronet modification moved the variable declarations here
//
char* L2_buffer_allocation_baseline;
char* L2_buffer_allocation_end_baseline;
//dronet modification added pointers to buffer allocation

/* Moves the weights and the biases from hyperflash to hyperram */
int network_setup()
{
  pi_task_t task = {0};
  pi_task_block(&task);
  struct pi_device fs;
  struct pi_device flash;
  pi_hyperram_conf_init(&ram_conf);
  open_filesystem(&flash, &fs);
  pi_open_from_conf(&ram, &ram_conf);
  pi_ram_open(&ram);
  pi_fs_file_t *file;
  pi_ram_alloc(&ram, &L3_weights, (uint32_t) 4800000);
  pi_ram_alloc(&ram, &L3_input, (uint32_t) 1500000);
  pi_ram_alloc(&ram, &L3_output, (uint32_t) 1500000);
#ifdef VERBOSE
    printf("\nL3 Buffer alloc initial\t@ %d:\t%s\n", (unsigned int)L3_weights, L3_weights?"Ok":"Failed");
    printf("\nL3 Buffer alloc initial\t@ %d:\t%s\n", (unsigned int)L3_input, L3_input?"Ok":"Failed");
    printf("\nL3 Buffer alloc initial\t@ %d:\t%s\n", (unsigned int)L3_output, L3_output?"Ok":"Failed");
#endif
  unsigned int rdDone = 0;
  for (int i=0;i<14;i++)
  {
    file = pi_fs_open(&fs, L3_weights_files[i], 0);
    if (file == NULL)
    {
      printf("file open failed\n");
      return -1;
    }
    L3_weights_size[i] = file->size + rdDone;
    int flashBuffSize = FLASH_BUFF_SIZE * sizeof(char);
    while(rdDone < (L3_weights_size[i] / sizeof(char)))
    {
      int size = pi_fs_read(file, flashBuffer, flashBuffSize);
      pi_ram_write(&ram, L3_weights+rdDone, flashBuffer,size);
      rdDone += size / sizeof(char);
    }
  }
  file = pi_fs_open(&fs, "inputs.hex", 0);
  if (file == NULL)
  {
    printf("file open failed\n");
    return -1;
  }
  activations_input = L3_weights+rdDone;
  rdDone = 0;
  int flashBuffSize = FLASH_BUFF_SIZE * sizeof(char);
  // loop on chunk in file
  while(rdDone < (40000 / sizeof(char)))
  {
    // read from HyperFlash
    int size = pi_fs_read(file, flashBuffer, flashBuffSize);
    // write to HyperRam
    pi_ram_write(&ram, activations_input+rdDone, flashBuffer, (uint32_t) size);
    rdDone += size / sizeof(char);
  }


  // Allocate L2 memory once-for-all
  L2_buffer_allocation = (char*) pmsis_l2_malloc(410000);
  L2_buffer_tofree_copy = L2_buffer_allocation;
  L2_buffer_allocation_end = L2_buffer_allocation + 410000;
  // Store baseline addresses. Needed in the while loop, at the beginning of each new inference
  L2_buffer_allocation_baseline = L2_buffer_allocation;
  L2_buffer_allocation_end_baseline = L2_buffer_allocation_end;
  // Return L2 buffer. We use this space to write images captured by the camera
  return L2_buffer_allocation;
  //dronet modification: returning pointer to the allocated space
}

// on cluster function execution
void cluster_main(void *arg)
{
  int *real_arg = (int *) arg;
  network_run((unsigned int) real_arg[0]);
}

// parallelization of the function given the number of cores
void pulp_parallel(void *arg)
{
  pi_cl_team_fork(NUM_CORES, (void *)cluster_main, arg);
}

void network_run_FabricController()
{
  int arg[1];
  arg[0] = (unsigned int) L3_weights_size;
  PMU_set_voltage(1000, 0);
  pi_time_wait_us(10000);
  pi_freq_set(PI_FREQ_DOMAIN_FC, 100000000);
  pi_time_wait_us(10000);
  pi_freq_set(PI_FREQ_DOMAIN_CL, 100000000);
  pi_time_wait_us(10000);

  struct pi_device cluster_dev = {0};
  struct pi_cluster_conf conf;
  struct pi_cluster_task cluster_task = {0};
  // task parameters allocation
  pi_cluster_task(&cluster_task, pulp_parallel, arg);
  cluster_task.stack_size = 4096;
  cluster_task.slave_stack_size = 3072;
  // First open the cluster
  pi_cluster_conf_init(&conf);
  conf.id=0;
  pi_open_from_conf(&cluster_dev, &conf);
  if (pi_cluster_open(&cluster_dev))
    return -1;
  // Then offload an entry point, this will get executed on the cluster controller
  pi_cluster_send_task_to_cl(&cluster_dev, &cluster_task);
  // closing of the cluster
  pi_cluster_close(&cluster_dev);
}

//dronet modification: here we had the variable declarations that were moved
//higher

void network_run(unsigned int L3_weights_size)
{

/*
  - initial buffer allocation L2 and L1
  - variable declaration
*/
/* ---------------------------------- */
/* -------- SECTION 0 BEGIN --------- */
/* ---------------------------------- */
  uint16_t out_mult = 0;
  uint16_t out_shift = 0;
  uint16_t inmul1 = 0;
  uint16_t inmul2 = 0;
  int branch_active = 0;
  int branch_keep_active = 0;
  int counter = 0;
  int counter_keep = 0;
  int valid = 0;
  static int keeping = 0;
  static int activation_to_keep_delloced = 0;
  int branch_output_index = 0;
  static int keep_index = 0;
  bypass_activations = 0;
  activation_to_keep = 0;
  int bypass_dimension = 0;
  int bypass_to_dealloc = 0;
  int activation_dimension = 0;
  int d_buffering_weights_t = 0;
  int error_presence = 0;
  int bypass_side = 0;
  int bypass_used_as_out = 0;
  int input_used_as_out = 0;
  int valid_keep = 0;
  int bypass_side_keep = 0;
  int d_buffering_weights_e = 0;
  int d_buffering_inputs = 0;
  int d_buffering_outputs = 0;
  int begin_end_n = 1;
  pi_cl_ram_req_t buff_req1;
  L3_weights_internal = L3_weights;
  transfer_weights = d_buffering_weights_t ? L2_weights_2 : L2_weights_1;
  exec_weights = d_buffering_weights_e ? L2_weights_2 : L2_weights_1;
  bypass_weights = d_buffering_weights_e ? L2_weights_2 : L2_weights_1;
  pi_cl_alloc_req_t alloc_req = {0};
  pi_cl_free_req_t free_req = {0};
  if (pi_core_id()==0)
  {
    // Restore original addresses
    L2_buffer_allocation = L2_buffer_allocation_baseline;
    L2_buffer_allocation_end = L2_buffer_allocation_end_baseline;
    // Allocate L1 buffer

    l1_buffer = pmsis_l1_malloc((uint32_t) 38000);
#ifdef VERBOSE
    printf("\nL2 Buffer alloc initial\t@ 0x%08x:\t%s\n", (unsigned int)L2_buffer_allocation, L2_buffer_allocation?"Ok":"Failed");
    printf("L1 Buffer alloc initial\t@ 0x%08x:\t%s\n\n", (unsigned int)l1_buffer, l1_buffer?"Ok":"Failed");
#endif
  }
/* ---------------------------------- */
/* --------- SECTION 0 END ---------- */
/* ---------------------------------- */

/*
  - initial copies from L3 of input
  - copies of weights of first 2 layers
*/
/* ---------------------------------- */
/* -------- SECTION 1 BEGIN --------- */
/* ---------------------------------- */
  if(pi_core_id()==0)
  {
/*
  - input allocation and copy
*/
    dory_L2_alloc(&L2_buffer_allocation,
      &L2_buffer_allocation_end,
      &L2_input,
      80000, // dronet modification: multiplied allocation by 2
      begin_end_n // begin is 1, end is 0
      );
    //pi_cl_ram_read(&ram, activations_input, L2_input, 40000, &buff_req1);
    //pi_cl_ram_read_wait(&buff_req1);
    //dronet modification: commented out the two lines above
/*
  - first layer weights allocation and copy
*/
    dory_L2_alloc(&L2_buffer_allocation,
      &L2_buffer_allocation_end,
      &L2_weights_1,
      164,
      begin_end_n // begin is 1, end is 0
      );
    begin_end_n = !begin_end_n;
    transfer_weights = L2_weights_1;
    exec_weights = L2_weights_1;
    pi_cl_ram_read(&ram, L3_weights_internal, transfer_weights, 164, &buff_req1);
    pi_cl_ram_read_wait(&buff_req1);
/*
  - output of the first layer allocation
*/
    dory_L2_alloc(&L2_buffer_allocation,
      &L2_buffer_allocation_end,
      &L2_output,
      40000,
      begin_end_n // begin is 1, end is 0
      );
    begin_end_n = !begin_end_n;
  }
/* ---------------------------------- */
/* --------- SECTION 1 END ---------- */
/* ---------------------------------- */
/* MAIN SECTION
  - for loop over all the layers of the network
  - double buffering using L3
  - check on layers to be executed from L3
  - residual check at the end of each layer
*/
/* ---------------------------------- */
/* -------- SECTION 2 BEGIN --------- */
/* ---------------------------------- */
  for(int i = 0; i < 15; i++)
  {
    if(pi_core_id()==0)
    {
      // copy of weights of next layers:
      // 1. copy only if we have to allocate the weights (hence not weights tiled from L3 and not pooling/add layer)
      // 2. waits before the read if we want to implement a double buffering, after if not.
      // Waiting based on the fact if layer need or not transfers from L3 memory.
      if(i < 14)
      {
        if (allocate_layer[i+1] == 1)
        {
          if (L3_layers[i-1] == 0 && i > 0)
            pi_cl_ram_read_wait(&buff_req1);
          pi_cl_ram_read(&ram, L3_weights_internal + cumulative_weights_dimension[i+1], transfer_weights, check_weights_dimension[i+1], &buff_req1);
          if (L3_layers[i] == 1)
            pi_cl_ram_read_wait(&buff_req1);
        }
      }
    }

    out_mult = out_mult_vector[i];
    out_shift = out_shift_vector[i];
    inmul1 = inmul1_vector[i];
    inmul2 = inmul2_vector[i];
    pi_cl_team_barrier(0);
    unsigned int args[13] = {L3_input,
      L3_output,
      L3_weights_internal + cumulative_weights_dimension[i],
      L2_input,
      bypass_activations,
      L2_output,
      exec_weights,
      l1_buffer,
      &ram,
      out_mult,
      inmul1,
      inmul2,
      out_shift};
    if (branch_change[i-1] == 1 && branch_input[i] == 0)
    {
      args[0] = bypass_L3_input;
      args[1] = bypass_L3_output;
      args[3] = bypass_activations;
    }
    if(branch_input[i] == 1 && keeping == 1)
    {
      args[4] = activation_to_keep;
    }
    switch (i)
    {
      case 0:
        layerConvBNRelu0(args);
        break;
      case 1:
        layerMaxPool1(args);
        break;
      case 2:
        layerConvDWBNRelu2(args);
        break;
      case 3:
        layerConvBNRelu3(args);
        break;
      case 4:
        layerConvDWBNRelu4(args);
        break;
      case 5:
        layerConvBNRelu3(args);
        break;
      case 6:
        layerConvDWBNRelu6(args);
        break;
      case 7:
        layerConvBNRelu7(args);
        break;
      case 8:
        layerConvDWBNRelu8(args);
        break;
      case 9:
        layerConvBNRelu9(args);
        break;
      case 10:
        layerConvDWBNRelu10(args);
        break;
      case 11:
        layerConvBNRelu11(args);
        break;
      case 12:
        layerConvDWBNRelu12(args);
        break;
      case 13:
        layerConvBNRelu13(args);
        break;
      case 14:
        layerMatMul14_last(args);
        break;
    }
    pi_cl_team_barrier(0);

    // dronet modification: CNN OUTPUTS
    if (i==14 && pi_core_id()==0){ //last iteration, core#0
      // Steering
      int32_t angle = *(int32_t*)(L2_output);
      // Collision
      int32_t prob_of_col = *(int32_t*)(L2_output+4);
      // Output variable
      ResOut[0] = angle;
      ResOut[1] = prob_of_col;
#ifdef DEBUG_PRINT
      // Print CNN outputs
      printf("network.c: Steering Angle: %d, Collision: %d \n",  angle, prob_of_col);
#endif
    }

    // prevents error from compiler
    if (pi_core_id()==0)
    {
      asm volatile("": : :"memory");
      unsigned int temp = L3_input;
      L3_input = L3_output;
      asm volatile("": : :"memory");
      L3_output = temp;
      asm volatile("": : :"memory");
    }

#ifdef VERBOSE
    if(pi_core_id()==0)
    {
      printf("Layer %d ended: \n", i);
    }
#endif
    if(branch_change[i] == 1)
    {
      keep_index = i;
    }
    if (i < 14)
    {
      if(pi_core_id()==0)
      {
        if (branch_input[i] == 1)
        {
          valid = 1;
          valid_keep = 1;
        }

        // deallocation of weights
        if (layer_with_weights[i] == 1)
          dory_L2_free(&L2_buffer_allocation,
            &L2_buffer_allocation_end,
            check_weights_dimension[i],
            begin_end_n // begin is 1, end is 0
            );
        if (layer_with_weights[i+1] == 1)
        {
          d_buffering_weights_e = !d_buffering_weights_e;
          exec_weights = d_buffering_weights_e ? L2_weights_2 : L2_weights_1;
        }
        // deallocation of input if not part of a residual connection
        //IT CAN NOT WORK FOR SOME CASES!!!
        if ((branch_output[i-1] !=1 && branch_change[i-1] != 1) && input_used_as_out!=1 || i==0)
        {
          dory_L2_free(&L2_buffer_allocation,
            &L2_buffer_allocation_end,
            check_activations_dimension[i],
            begin_end_n // begin is 1, end is 0
            );

        }

        // deallocation of a residual activation previously stored
        if(valid_keep == 1 && keeping == 1 && bypass_side_keep==begin_end_n && bypass_used_as_out!=1)
        {
          dory_L2_free(&L2_buffer_allocation,
            &L2_buffer_allocation_end,
            activation_dimension,
            begin_end_n // begin is 1, end is 0
            );
          counter_keep = 0;
          branch_keep_active = 0;
          keeping = 0;
          activation_to_keep_delloced = 0;
        }
        // MUST MAKE SURE THAT ACTIVATION_TO_KEEP IS NOT INFRONT OF BYPASS AND THAT IT IS
        // SAFE TO DEALLOC BYPASS ACTIVATION. IT'S MOST LIKELY ONLY DONE WHEN ON ADD LAYER
        if (branch_input[i]==1 && bypass_to_dealloc == 1)
        {
          dory_L2_free(&L2_buffer_allocation,
            &L2_buffer_allocation_end,
            bypass_dimension,
            begin_end_n // begin is 1, end is 0
            );
          counter = 0;
          branch_active = 0;
          bypass_to_dealloc = 0;
        }
        // Keep last layer of left side until add layer is encountered.
        if (branch_change[i] == 1 && branch_output[i] == 0 && branch_last[i] == 0)
        {
          activation_to_keep = L2_output;
          activation_dimension = check_activations_out_dimension[i];
          keeping = 1;
          branch_keep_active = 1;
          activation_to_keep_delloced = 1;
          bypass_side_keep = !begin_end_n;
          valid_keep = 0;
        }
        if (branch_output[i] == 1)
        {
          bypass_L3_input = L3_input;
          bypass_L3_output = L3_output;
          branch_output_index = i;
          bypass_activations = L2_output;
          bypass_dimension = check_activations_out_dimension[i];
          branch_active = 1;
          bypass_to_dealloc = 1;
          bypass_side = !begin_end_n;
          valid = 0;
        }
        L2_input = L2_output;
        // allocation of output feature space
        if (branch_input[i+1]!=1 || (branch_input[i+1]==1 && bypass_side != begin_end_n && keeping == 0))
        {
          dory_L2_alloc(&L2_buffer_allocation,
            &L2_buffer_allocation_end,
            &L2_output,
            check_activations_out_dimension[i+1],
            begin_end_n // begin is 1, end is 0
            );
          input_used_as_out = 0;
          bypass_used_as_out = 0;
        }
        else if (keeping == 1)
        {
          if (bypass_side_keep == begin_end_n)
          {
            L2_output = L2_input;
            input_used_as_out = 1;
          }
          else
          {
            L2_output = activation_to_keep;
            keeping = 0;
          }
        }
        else
        {
          L2_output = bypass_activations;
          bypass_used_as_out = 1;
          bypass_to_dealloc = 0;
        }
        if (i < 13)
        {
          if (branch_input[i+1]==1 && bypass_side_keep == begin_end_n && keeping==1)
            begin_end_n = !begin_end_n;
          // allocation of weights for next next layer, if necessary.
          if (layer_with_weights[i+2] == 1)
          {
            if (d_buffering_weights_e==1)
            {
              dory_L2_alloc(&L2_buffer_allocation,
                &L2_buffer_allocation_end,
                &L2_weights_1,
                check_weights_dimension[i+2],
                begin_end_n // begin is 1, end is 0
                );
            }
            else
            {
              dory_L2_alloc(&L2_buffer_allocation,
                &L2_buffer_allocation_end,
                &L2_weights_2,
                check_weights_dimension[i+2],
                begin_end_n // begin is 1, end is 0
                );
            }
            d_buffering_weights_t = !d_buffering_weights_t;
            transfer_weights = d_buffering_weights_t ? L2_weights_2 : L2_weights_1;
          }
        }
        //switching output and input in the buffer for allocation.
        begin_end_n = !begin_end_n;
      }

    }
  }
/* ---------------------------------- */
/* --------- SECTION 2 END ---------- */
/* ---------------------------------- */

/* ---------------------------------- */
/* -------- SECTION 3 BEGIN --------- */
/* ---------------------------------- */


  if (pi_core_id()==0)
  {
    //pi_cl_l2_free(L2_buffer_tofree_copy, (uint32_t) 420000, &free_req);
    //pi_cl_l2_free_wait(&free_req);
    //dronet modification: commented the two lines above
    pmsis_l1_malloc_free(l1_buffer, (uint32_t) 38000 );
  }
/* ---------------------------------- */
/* --------- SECTION 3 END ---------- */
/* ---------------------------------- */
}

