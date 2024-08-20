#!/bin/sh

####################################################
############# 1. TRAINING / VALIDATION #############
####################################################

# ResBlock with bypass
python training.py --gpu=0 --model_weights_path=resblock_1.0    --block_type=ResBlock   --depth_mult=1.0    --bypass=True
python training.py --gpu=0 --model_weights_path=resblock_0.5    --block_type=ResBlock   --depth_mult=0.5    --bypass=True
python training.py --gpu=0 --model_weights_path=resblock_0.25   --block_type=ResBlock   --depth_mult=0.25   --bypass=True
python training.py --gpu=0 --model_weights_path=resblock_0.125  --block_type=ResBlock   --depth_mult=0.125  --bypass=True

# ResBlock without bypass
python training.py --gpu=0 --model_weights_path=resblock_noby_1.0     --block_type=ResBlock   --depth_mult=1.0    --bypass=False
python training.py --gpu=0 --model_weights_path=resblock_noby_0.5     --block_type=ResBlock   --depth_mult=0.5    --bypass=False
python training.py --gpu=0 --model_weights_path=resblock_noby_0.25    --block_type=ResBlock   --depth_mult=0.25   --bypass=False
python training.py --gpu=0 --model_weights_path=resblock_noby_0.125   --block_type=ResBlock   --depth_mult=0.125  --bypass=False

# DW+PW with with bypass
python training.py --gpu=0 --model_weights_path=dw_pw_1.0       --block_type=Depthwise   --depth_mult=1.0    --bypass=True
python training.py --gpu=0 --model_weights_path=dw_pw_0.5       --block_type=Depthwise   --depth_mult=0.5    --bypass=True
python training.py --gpu=0 --model_weights_path=dw_pw_0.25      --block_type=Depthwise   --depth_mult=0.25   --bypass=True
python training.py --gpu=0 --model_weights_path=dw_pw_0.125     --block_type=Depthwise   --depth_mult=0.125  --bypass=True

# DW+PW without bypass
python training.py --gpu=0 --model_weights_path=dw_pw_noby_1.0   --block_type=Depthwise  --depth_mult=1.0    --bypass=False
python training.py --gpu=0 --model_weights_path=dw_pw_noby_0.5   --block_type=Depthwise  --depth_mult=0.5    --bypass=False
python training.py --gpu=0 --model_weights_path=dw_pw_noby_0.25  --block_type=Depthwise  --depth_mult=0.25   --bypass=False
python training.py --gpu=0 --model_weights_path=dw_pw_noby_0.125 --block_type=Depthwise  --depth_mult=0.125  --bypass=False

# IRLB with bypass
python training.py --gpu=0 --model_weights_path=irlb_1.0       --block_type=IRLB   --depth_mult=1.0    --bypass=True
python training.py --gpu=0 --model_weights_path=irlb_0.5       --block_type=IRLB   --depth_mult=0.5    --bypass=True
python training.py --gpu=0 --model_weights_path=irlb_0.25      --block_type=IRLB   --depth_mult=0.25   --bypass=True
python training.py --gpu=0 --model_weights_path=irlb_0.125     --block_type=IRLB   --depth_mult=0.125  --bypass=True

# IRLB without bypass
python training.py --gpu=0 --model_weights_path=irlb_noby_1.0   --block_type=IRLB  --depth_mult=1.0    --bypass=False
python training.py --gpu=0 --model_weights_path=irlb_noby_0.5   --block_type=IRLB  --depth_mult=0.5    --bypass=False
python training.py --gpu=0 --model_weights_path=irlb_noby_0.25  --block_type=IRLB  --depth_mult=0.25   --bypass=False
python training.py --gpu=0 --model_weights_path=irlb_noby_0.125 --block_type=IRLB  --depth_mult=0.125  --bypass=False

## TESTING
# pulp dronet v3
python testing.py --gpu=0 --model_weights_path=./model/pulp-dronet-v3-resblock-1.0.pth      --block_type=ResBlock  --depth_mult=1.0    --bypass=True  --data_path_testing=./dataset/
# tiny pulp dronet v3
python testing.py --gpu=0 --model_weights_path=./model/tiny-pulp-dronet-v3-dw-pw-0.125.pth  --block_type=Depthwise  --depth_mult=0.125 --bypass=False --data_path_testing=./dataset/


####################################################
################# 2. Quantization ##################
####################################################

# pulp dronet v3
python quantize.py --gpu=0 --model_weights_path=../model/pulp-dronet-v3-resblock-1.0.pth        --block_type=ResBlock   --depth_mult=1.0   --bypass=True  --data_path_testing=./dataset/
# tiny pulp dronet v3
python quantize.py --gpu=0 --model_weights_path=../model/tiny-pulp-dronet-v3-dw-pw-0.125.pth    --block_type=Depthwise  --depth_mult=0.125 --bypass=False --data_path_testing=./dataset/

####################################################
################# 4. DORY GENERATION ###############
####################################################

# GAP SDK
conda activate gap_sdk
source <YOUR_PATH_HERE>/gap_sdk/configs/ai_deck.sh
export GAPY_OPENOCD_CABLE=<YOUR_PATH_HERE>/gap_sdk/tools/gap8-openocd/tcl/interface/ftdi/olimex-arm-usb-ocd-h.cfg

# go to dory directory
cd nemo-dory/dory/dory_examples/

# generate pulp-dronet v3
python network_generate.py --network_dir=../../nemo_output/pulp-dronet-v3-resblock-1.0/ --sdk=gap_sdk --Bn_Relu_Bits=64 --l2_buffer_size 410000  --l1_buffer_size 35000 --verbose_level=Check_all+Perf_final
mv application/ application-pulp-dronet-v3 && cd application-pulp-dronet-v3
make clean all run CORE=8   # only run code
make clean all flash CORE=8 # flash code

# generate tiny-pulp-dronet v3
python network_generate.py --network_dir=../../nemo_output/tiny-pulp-dronet-v3-dw-pw-0.125/ --sdk=gap_sdk --Bn_Relu_Bits=64 --l2_buffer_size 410000  --l1_buffer_size 35000 --verbose_level=Check_all+Perf_final
mv application/ application-tiny-pulp-dronet-v3/ && cd application-tiny-pulp-dronet-v3
make clean all run CORE=8   # only run code
make clean all flash CORE=8 # flash code