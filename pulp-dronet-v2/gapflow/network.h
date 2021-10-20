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
                                                                               
 File:    network.h   
 Authors: Vlad Niculescu   	<vladn@iis.ee.ethz.ch>
          Lorenzo Lamberti 	<lorenzo.lamberti@unibo.it>
          Daniele Palossi  <dpalossi@iis.ee.ethz.ch> <daniele.palossi@idsia.ch>
 Date:    15.03.2021                                                           
-------------------------------------------------------------------------------*/


#ifndef __NETWORK_H__
#define __NETWORK_H__

#include "Gap.h"
#define __PREFIX(x) network ## x
extern AT_HYPERFLASH_FS_EXT_ADDR_TYPE __PREFIX(_L3_Flash);
// char AT_GraphNodeNames[16] = {'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p'};

#endif
