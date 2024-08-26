/*-----------------------------------------------------------------------------
 Copyright (C) 2024 University of Bologna, Italy, ETH Zurich, Switzerland.
 All rights reserved.

 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 See LICENSE in the top directory for details.
 You may obtain a copy of the License at

   http://www.apache.org/licenses/LICENSE-2.0

 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.

 File:    uart_dma_pulp.h
 Author:  Lorenzo Lamberti      <lorenzo.lamberti@unibo.it>
 Date:    01.03.2024
-------------------------------------------------------------------------------*/

#ifndef __UART_DMA_PULP_H
#define __UART_DMA_PULP_H

#ifdef __cplusplus
 extern "C" {
#endif

#include "stm32f4xx.h"

typedef enum {FAILED = 0, PASSED = !FAILED} TestStatus;


  /* Definition for USARTx resources ******************************************/
#define USARTx                           USART3
#define USARTx_CLK                       RCC_APB1Periph_USART3
#define USARTx_CLK_INIT                  RCC_APB1PeriphClockCmd
#define USARTx_IRQn                      USART3_IRQn
#define USARTx_IRQHandler                USART3_IRQHandler

#define USARTx_TX_PIN                    GPIO_Pin_10
#define USARTx_TX_GPIO_PORT              GPIOC
#define USARTx_TX_GPIO_CLK               RCC_AHB1Periph_GPIOC
#define USARTx_TX_SOURCE                 GPIO_PinSource10
#define USARTx_TX_AF                     GPIO_AF_USART3

#define USARTx_RX_PIN                    GPIO_Pin_11
#define USARTx_RX_GPIO_PORT              GPIOC
#define USARTx_RX_GPIO_CLK               RCC_AHB1Periph_GPIOC
#define USARTx_RX_SOURCE                 GPIO_PinSource11
#define USARTx_RX_AF                     GPIO_AF_USART3

  /* Definition for DMAx resources ********************************************/
#define USARTx_DR_ADDRESS                ((uint32_t)USART3 + 0x04)
#define USARTx_DMA                       DMA1
#define USARTx_DMAx_CLK                  RCC_AHB1Periph_DMA1

#define USARTx_TX_DMA_CHANNEL            DMA_Channel_4
#define USARTx_TX_DMA_STREAM             DMA1_Stream3
#define USARTx_TX_DMA_FLAG_FEIF          DMA_FLAG_FEIF3
#define USARTx_TX_DMA_FLAG_DMEIF         DMA_FLAG_DMEIF3
#define USARTx_TX_DMA_FLAG_TEIF          DMA_FLAG_TEIF3
#define USARTx_TX_DMA_FLAG_HTIF          DMA_FLAG_HTIF3
#define USARTx_TX_DMA_FLAG_TCIF          DMA_FLAG_TCIF3

#define USARTx_RX_DMA_CHANNEL            DMA_Channel_4
#define USARTx_RX_DMA_STREAM             DMA1_Stream1
#define USARTx_RX_DMA_FLAG_FEIF          DMA_FLAG_FEIF1
#define USARTx_RX_DMA_FLAG_DMEIF         DMA_FLAG_DMEIF1
#define USARTx_RX_DMA_FLAG_TEIF          DMA_FLAG_TEIF1
#define USARTx_RX_DMA_FLAG_HTIF          DMA_FLAG_HTIF1
#define USARTx_RX_DMA_FLAG_TCIF          DMA_FLAG_TCIF1

#define USARTx_DMA_TX_IRQn               DMA1_Stream3_IRQn
#define USARTx_DMA_RX_IRQn               DMA1_Stream1_IRQn
#define USARTx_DMA_TX_IRQHandler         DMA1_Stream3_IRQHandler
#define USARTx_DMA_RX_IRQHandler         DMA1_Stream1_IRQHandler

#define UART3_RX_DMA_ALL_FLAGS (DMA_FLAG_FEIF1 | DMA_FLAG_DMEIF1 | DMA_FLAG_TEIF1 | DMA_FLAG_HTIF1 | DMA_FLAG_TCIF1 )


#ifdef __cplusplus
}
#endif

#endif
