#!/bin/sh

#################################################
############# TRAINING / VALIDATION #############
#################################################

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
python testing.py --gpu=0 --model_weights_path=./model/pulp-dronet-v3-resblock-1.0.pth --block_type=ResBlock  --depth_mult=1.0    --bypass=True  --data_path_testing=/home/llamberti/work/opensource/pulp-dronet/tiny-pulp-dronet-v3/dataset/
#tiny pulp dronet v3
python testing.py --gpu=0 --model_weights_path=./model/new_dw_pw_noby_0.125_100.pth    --block_type=Depthwise  --depth_mult=0.125 --bypass=False --data_path_testing=/home/llamberti/work/opensource/pulp-dronet/tiny-pulp-dronet-v3/dataset/

#################################################
################# Quantization ##################
#################################################
# pulp dronet v3
python quantize.py --gpu=0 --model_weights_path=./model/pulp-dronet-v3-resblock-1.0.pth --block_type=ResBlock   --depth_mult=1.0   --bypass=True  --data_path_testing=/home/llamberti/work/opensource/pulp-dronet/tiny-pulp-dronet-v3/dataset/
#tiny pulp dronet v3
python quantize.py --gpu=0 --model_weights_path=./model/new_dw_pw_noby_0.125_100.pth    --block_type=Depthwise  --depth_mult=0.125 --bypass=False --data_path_testing=/home/llamberti/work/opensource/pulp-dronet/tiny-pulp-dronet-v3/dataset/
