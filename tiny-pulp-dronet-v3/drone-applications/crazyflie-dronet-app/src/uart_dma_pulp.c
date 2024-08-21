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

 File:    uart_dma_pulp.c
 Author:  Lorenzo Lamberti      <lorenzo.lamberti@unibo.it>
 Date:    01.03.2024
-------------------------------------------------------------------------------*/

#include "uart_dma_pulp.h"

DMA_InitTypeDef  DMA_InitStructure;

static void USART_Config(uint32_t baudrate, int8_t *pulpRxBuffer, uint32_t BUFFERSIZE);

void USART_DMA_Start(uint32_t baudrate, int8_t *pulpRxBuffer, uint32_t BUFFERSIZE)
{
  // Setup Communication
  USART_Config(baudrate, pulpRxBuffer, BUFFERSIZE);

  DMA_ITConfig(USARTx_RX_DMA_STREAM, DMA_IT_TC, ENABLE);

  // Enable DMA USART RX Stream
  DMA_Cmd(USARTx_RX_DMA_STREAM,ENABLE);

  // Enable USART DMA RX Requsts
  USART_DMACmd(USARTx, USART_DMAReq_Rx, ENABLE);

  // Clear DMA Transfer Complete Flags
  DMA_ClearFlag(USARTx_RX_DMA_STREAM,USARTx_RX_DMA_FLAG_TCIF);

  // Clear USART Transfer Complete Flags
  USART_ClearFlag(USARTx,USART_FLAG_TC);

  DMA_ClearFlag(USARTx_RX_DMA_STREAM, UART3_RX_DMA_ALL_FLAGS);
  NVIC_EnableIRQ(DMA1_Stream1_IRQn);
}

static void USART_Config(uint32_t baudrate, int8_t *pulpRxBuffer, uint32_t BUFFERSIZE)
{
    USART_InitTypeDef USART_InitStructure;
    GPIO_InitTypeDef GPIO_InitStructure;

    // Enable GPIO clock
    RCC_AHB1PeriphClockCmd(USARTx_TX_GPIO_CLK | USARTx_RX_GPIO_CLK, ENABLE);

    // Enable USART clock
    USARTx_CLK_INIT(USARTx_CLK, ENABLE);

    // Enable the DMA clock
    RCC_AHB1PeriphClockCmd(USARTx_DMAx_CLK, ENABLE);

    // Connect USART pins to Crazyflie RX1 annd TX1 - USART3 in the STM32 */
    GPIO_PinAFConfig(USARTx_TX_GPIO_PORT, USARTx_TX_SOURCE, USARTx_TX_AF);
    GPIO_PinAFConfig(USARTx_RX_GPIO_PORT, USARTx_RX_SOURCE, USARTx_RX_AF);

    // Configure USART Tx and Rx as alternate function push-pull
    GPIO_InitStructure.GPIO_Mode = GPIO_Mode_AF;
    GPIO_InitStructure.GPIO_Speed = GPIO_Speed_100MHz;
    GPIO_InitStructure.GPIO_OType = GPIO_OType_PP;
    GPIO_InitStructure.GPIO_PuPd = GPIO_PuPd_UP;

    GPIO_InitStructure.GPIO_Pin = USARTx_TX_PIN;
    GPIO_Init(USARTx_TX_GPIO_PORT, &GPIO_InitStructure);

    GPIO_InitStructure.GPIO_Pin = USARTx_RX_PIN;
    GPIO_Init(USARTx_RX_GPIO_PORT, &GPIO_InitStructure);

    // USARTx configuration
    USART_OverSampling8Cmd(USARTx, ENABLE);

    USART_InitStructure.USART_BaudRate = baudrate;
    USART_InitStructure.USART_WordLength = USART_WordLength_8b;
    USART_InitStructure.USART_StopBits = USART_StopBits_1;
    /* When using Parity the word length must be configured to 9 bits */
    USART_InitStructure.USART_Parity = USART_Parity_No;
    USART_InitStructure.USART_HardwareFlowControl = USART_HardwareFlowControl_None;
    USART_InitStructure.USART_Mode = USART_Mode_Rx;
    USART_Init(USARTx, &USART_InitStructure);

    /* Configure DMA Initialization Structure */
    DMA_InitStructure.DMA_BufferSize = BUFFERSIZE ;
    DMA_InitStructure.DMA_FIFOMode = DMA_FIFOMode_Disable ;
    DMA_InitStructure.DMA_FIFOThreshold = DMA_FIFOThreshold_1QuarterFull ;
    DMA_InitStructure.DMA_MemoryBurst = DMA_MemoryBurst_Single ;
    DMA_InitStructure.DMA_MemoryDataSize = DMA_MemoryDataSize_Byte;
    DMA_InitStructure.DMA_MemoryInc = DMA_MemoryInc_Enable;
    DMA_InitStructure.DMA_Mode = DMA_Mode_Circular;
    DMA_InitStructure.DMA_PeripheralBaseAddr =(uint32_t) (&(USARTx->DR)) ;
    DMA_InitStructure.DMA_PeripheralBurst = DMA_PeripheralBurst_Single;
    DMA_InitStructure.DMA_PeripheralDataSize = DMA_PeripheralDataSize_Byte;
    DMA_InitStructure.DMA_PeripheralInc = DMA_PeripheralInc_Disable;
    DMA_InitStructure.DMA_Priority = DMA_Priority_High;

    /* Configure RX DMA */
    DMA_InitStructure.DMA_Channel = USARTx_RX_DMA_CHANNEL ;
    DMA_InitStructure.DMA_DIR = DMA_DIR_PeripheralToMemory ;
    DMA_InitStructure.DMA_Memory0BaseAddr =(uint32_t)pulpRxBuffer ;
    DMA_Init(USARTx_RX_DMA_STREAM,&DMA_InitStructure);

    /* Enable USART */
    USART_Cmd(USARTx, ENABLE);
}
