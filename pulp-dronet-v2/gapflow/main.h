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
                                                                               
 File:    main.h 
 Authors: Vlad Niculescu   	<vladn@iis.ee.ethz.ch>
          Lorenzo Lamberti 	<lorenzo.lamberti@unibo.it>
          Daniele Palossi  <dpalossi@iis.ee.ethz.ch> <daniele.palossi@idsia.ch>
 Date:    15.03.2021                                                           
-------------------------------------------------------------------------------*/


#ifndef __IMAGENET_H__
#define __IMAGENET_H__

// #include "mobilenet_v1_1_0_224_quantKernels.h"

#include "Gap.h"
#include "gaplib/ImgIO.h"

#ifdef __EMUL__
#include <sys/types.h>
#include <unistd.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <sys/param.h>
#include <string.h>
#endif

#endif