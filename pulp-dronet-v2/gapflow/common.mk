# MODEL_PREFIX?=mobilenet_v1_1_0_224_quant
MODEL_PREFIX?=network
AT_INPUT_WIDTH?=200
AT_INPUT_HEIGHT?=200
AT_INPUT_COLORS?=1

## AT GENERATED NAMES
AT_CONSTRUCT = $(MODEL_PREFIX)CNN_Construct
AT_DESTRUCT = $(MODEL_PREFIX)CNN_Destruct
AT_CNN = $(MODEL_PREFIX)CNN
AT_L3_ADDR = $(MODEL_PREFIX)_L3_Flash
