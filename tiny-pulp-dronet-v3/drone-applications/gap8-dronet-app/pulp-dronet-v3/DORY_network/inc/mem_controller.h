/*
 * mem_controller.h
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

void dory_L2_alloc(unsigned int * L2_pointer_input_begin,
              unsigned int * L2_pointer_input_end,
              unsigned int * L2_pointer_output,
              int memory_to_allocate,
              int begin_end_n // begin is 1, end is 0
              );
void dory_L2_free(unsigned int * L2_pointer_input_begin,
            unsigned int * L2_pointer_input_end,
            int memory_to_free,
            int begin_end_n // begin is 1, end is 0
            );

void dory_L1_alloc(unsigned int * L2_pointer_input_begin,
              unsigned int * L2_pointer_output,
              int memory_to_allocate
              );


void dory_L1_free(unsigned int * L2_pointer_input_begin,
            int memory_to_free
            );