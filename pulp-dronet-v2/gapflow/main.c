/*-----------------------------------------------------------------------------
 Copyright (C) 2020-2021 ETH Zurich, Switzerland, University of Bologna, Italy.
 All rights reserved.                                                          
                                                                               
 Licensed under the Apache License, Version 2.0 (the "License");               
 you may not use this file except in compliance with the License.              
 See LICENSE.apache.md in the top directory for details.                       
 You may obtain a copy of the License at                                       
                                                                               
   http://www.apache.org/licenses/LICENSE-2.0                                  
                                                                               
 Unless required by applicable law or agreed to in writing, software           
 distributed under the License is distributed on an "AS IS" BASIS,             
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.      
 See the License for the specific language governing permissions and           
 limitations under the License.                                                
                                                                               
 File:    main.c   
 Authors: Vlad Niculescu   	<vladn@iis.ee.ethz.ch>
          Lorenzo Lamberti 	<lorenzo.lamberti@unibo.it>
          Daniele Palossi  <dpalossi@iis.ee.ethz.ch> <daniele.palossi@idsia.ch>
 Date:    15.03.2021                                                           
-------------------------------------------------------------------------------*/


#include "pmsis.h"
#include "bsp/flash/hyperflash.h"
#include "bsp/bsp.h"
#include "bsp/buffer.h"
#include "bsp/camera/himax.h"
#include "bsp/ram.h"
#include "bsp/ram/hyperram.h"
#include "bsp/display/ili9341.h"
#include "bsp/fs.h"
#include "bsp/flash/hyperflash.h"
#include "bsp/fs/readfs.h"

#include "main.h"
#include "networkKernels.h"


/* Defines */
#define NUM_CLASSES 	2
#define AT_INPUT_SIZE 	(AT_INPUT_WIDTH*AT_INPUT_HEIGHT*AT_INPUT_COLORS)

#define __XSTR(__s) __STR(__s)
#define __STR(__s) #__s 

typedef signed char NETWORK_OUT_TYPE;

// Global Variables
struct pi_device camera;
static pi_buffer_t buffer;
struct pi_device HyperRam;

L2_MEM NETWORK_OUT_TYPE *ResOut;
static uint32_t l3_buff;
AT_HYPERFLASH_FS_EXT_ADDR_TYPE AT_L3_ADDR = 0;


static void RunNetwork()
{
  printf("Running on cluster\n");
  printf("Start timer\n");
  gap_cl_starttimer();
  gap_cl_resethwtimer();
  AT_CNN(l3_buff, &ResOut[0], &ResOut[1]);
  printf("Runner completed\n");
}


int body(void)
{
	// Voltage-Frequency settings
	uint32_t voltage =1200;
	pi_freq_set(PI_FREQ_DOMAIN_FC, FREQ_FC*1000*1000);
	pi_freq_set(PI_FREQ_DOMAIN_CL, FREQ_CL*1000*1000);
	PMU_set_voltage(voltage, 0);
	printf("Set VDD voltage as %.2f, FC Frequency as %d MHz, CL Frequency = %d MHz\n", 
		(float)voltage/1000, FREQ_FC, FREQ_CL);

    pi_fs_file_t *file;
    struct pi_device fs;
    struct pi_device flash;
    struct pi_hyperflash_conf flash_conf;
    struct pi_readfs_conf conf0;

    pi_hyperflash_conf_init(&flash_conf);
    pi_open_from_conf(&flash, &flash_conf);
    if (pi_flash_open(&flash))
    {
        printf("Error flash open ! \n");
        pmsis_exit(-3);
    }

    // Open filesystem on flash
    pi_readfs_conf_init(&conf0);
    conf0.fs.flash = &flash;
    pi_open_from_conf(&fs, &conf0);
    if (pi_fs_mount(&fs))
    {
        printf("Error FS mounting ! \n");
        pmsis_exit(-2);
    }

	// Initialize the ram 
  	struct pi_hyperram_conf hyper_conf;
  	pi_hyperram_conf_init(&hyper_conf);
  	pi_open_from_conf(&HyperRam, &hyper_conf);
	if (pi_ram_open(&HyperRam))
	{
		printf("Error ram open !\n");
		pmsis_exit(-3);
	}

	// Allocate L3 buffer to store input data 
	if (pi_ram_alloc(&HyperRam, &l3_buff, (uint32_t) AT_INPUT_SIZE))
	{
		printf("Ram malloc failed !\n");
		pmsis_exit(-4);
	}

	// Allocate temp buffer for image data
	uint8_t* Input_1 = (uint8_t*) pmsis_l2_malloc(AT_INPUT_SIZE*sizeof(char));
	if(!Input_1){
		printf("Failed allocation!\n");
		pmsis_exit(1);
	}

	char *ImageName = __XSTR(AT_IMAGE);
	printf("Reading image from %s\n",ImageName);

	// Read image
	img_io_out_t type = IMGIO_OUTPUT_CHAR;
	if (ReadImageFromFile(ImageName, AT_INPUT_WIDTH, AT_INPUT_HEIGHT, AT_INPUT_COLORS, Input_1, AT_INPUT_SIZE*sizeof(char), type, 0)) {
		printf("Failed to load image %s\n", ImageName);
		pmsis_exit(-1);
	}
	printf("Finished reading image %s\n", ImageName);

	// Write greyscale image to RAM
	pi_ram_write(&HyperRam, (l3_buff), Input_1, (uint32_t) AT_INPUT_SIZE);
	pmsis_l2_malloc_free(Input_1, AT_INPUT_SIZE*sizeof(char));

	// Open the cluster
	struct pi_device cluster_dev;
	struct pi_cluster_conf conf;
	pi_cluster_conf_init(&conf);
	pi_open_from_conf(&cluster_dev, (void *)&conf);
	pi_cluster_open(&cluster_dev);

	// Task setup
	struct pi_cluster_task *task = pmsis_l2_malloc(sizeof(struct pi_cluster_task));
	if(task==NULL) {
	  printf("pi_cluster_task alloc Error!\n");
	  pmsis_exit(-1);
	}
	printf("Stack size is %d and %d\n",STACK_SIZE,SLAVE_STACK_SIZE );
	memset(task, 0, sizeof(struct pi_cluster_task));
	task->entry = &RunNetwork;
	task->stack_size = STACK_SIZE;
	task->slave_stack_size = SLAVE_STACK_SIZE;
	task->arg = NULL;

	// Allocate the output tensor
	ResOut = (NETWORK_OUT_TYPE *) AT_L2_ALLOC(0, NUM_CLASSES*sizeof(NETWORK_OUT_TYPE));
	if (ResOut==0) {
		printf("Failed to allocate Memory for Result (%ld bytes)\n", 2*sizeof(char));
		return 1;
	}

	// Network Constructor
	int err_const = AT_CONSTRUCT();
	if (err_const)
	{
	  printf("Graph constructor exited with error: %d\n", err_const);
	  return 1;
	}
	printf("Network Constructor was OK!\n");

	// Dispatch task on the cluster 
	pi_cluster_send_task_to_cl(&cluster_dev, task);

    printf("Model:\t%s\n\n", __XSTR(AT_MODEL_PREFIX));
	double out1 = 0.2460539 * (double)ResOut[0];
	double out2 = 0.00787402 * (double)ResOut[1];

	printf("With quantization: \n");
	printf("Output 1:\t%.6f\n", out1);
	printf("Output 2:\t%.6f\n", out2);

	// Performance counters
	unsigned int TotalCycles = 0, TotalOper = 0;
	printf("\n");
	for (int i=0; i<(sizeof(AT_GraphPerf)/sizeof(unsigned int)); i++) {
		printf("%45s: Cycles: %10d, Operations: %10d, Operations/Cycle: %f\n", AT_GraphNodeNames[i], AT_GraphPerf[i], AT_GraphOperInfosNames[i], ((float) AT_GraphOperInfosNames[i])/ AT_GraphPerf[i]);
		TotalCycles += AT_GraphPerf[i]; TotalOper += AT_GraphOperInfosNames[i];
	}
	printf("\n");
	printf("\t\t\t %s: Cycles: %10d, Operations: %10d, Operations/Cycle: %f\n", "Total", TotalCycles, TotalOper, ((float) TotalOper)/ TotalCycles);
	printf("\n");

	// Netwrok Destructor
	AT_DESTRUCT();
	pmsis_exit(0);
	return 0;
}


int main(void)
{
    printf("\n\n\t *** ImageNet classification on GAP ***\n");
    return pmsis_kickoff((void *) body);
}

