#!/bin/sh
# ResBlock with bypass
python training.py --gpu=0 --block_type=ResBlock --model_name=resblock_1.0    --depth_mult=1.0    --arch=dronet_dory --data_path=/home/lamberti/work/dataset/Dataset-PULP-Dronet-V3_aug/  --data_path_testing=/home/lamberti/work/dataset/temp/Dataset-PULP-Dronet-V3
python training.py --gpu=0 --block_type=ResBlock --model_name=resblock_0.5    --depth_mult=0.5    --arch=dronet_dory --data_path=/home/lamberti/work/dataset/Dataset-PULP-Dronet-V3_aug/  --data_path_testing=/home/lamberti/work/dataset/temp/Dataset-PULP-Dronet-V3
python training.py --gpu=0 --block_type=ResBlock --model_name=resblock_0.25   --depth_mult=0.25   --arch=dronet_dory --data_path=/home/lamberti/work/dataset/Dataset-PULP-Dronet-V3_aug/  --data_path_testing=/home/lamberti/work/dataset/temp/Dataset-PULP-Dronet-V3
python training.py --gpu=0 --block_type=ResBlock --model_name=resblock_0.125  --depth_mult=0.125  --arch=dronet_dory --data_path=/home/lamberti/work/dataset/Dataset-PULP-Dronet-V3_aug/  --data_path_testing=/home/lamberti/work/dataset/temp/Dataset-PULP-Dronet-V3

# DW+PW with no bypass
python training.py --gpu=0 --block_type=Depthwise --model_name=dw_pw_noby_0.125   --depth_mult=0.125  --arch=dronet_dory_no_residuals --data_path=/home/lamberti/work/dataset/Dataset-PULP-Dronet-V3_aug/  --data_path_testing=/home/lamberti/work/dataset/temp/Dataset-PULP-Dronet-V3
python training.py --gpu=0 --block_type=Depthwise --model_name=dw_pw_noby_0.25    --depth_mult=0.25   --arch=dronet_dory_no_residuals --data_path=/home/lamberti/work/dataset/Dataset-PULP-Dronet-V3_aug/  --data_path_testing=/home/lamberti/work/dataset/temp/Dataset-PULP-Dronet-V3
python training.py --gpu=0 --block_type=Depthwise --model_name=dw_pw_noby_0.5     --depth_mult=0.5    --arch=dronet_dory_no_residuals --data_path=/home/lamberti/work/dataset/Dataset-PULP-Dronet-V3_aug/  --data_path_testing=/home/lamberti/work/dataset/temp/Dataset-PULP-Dronet-V3
python training.py --gpu=0 --block_type=Depthwise --model_name=dw_pw_noby_1.0     --depth_mult=1.0    --arch=dronet_dory_no_residuals --data_path=/home/lamberti/work/dataset/Dataset-PULP-Dronet-V3_aug/  --data_path_testing=/home/lamberti/work/dataset/temp/Dataset-PULP-Dronet-V3
python training.py --gpu=0 --block_type=Depthwise --model_name=dw_pw_noby_2.0     --depth_mult=2.0    --arch=dronet_dory_no_residuals --data_path=/home/lamberti/work/dataset/Dataset-PULP-Dronet-V3_aug/  --data_path_testing=/home/lamberti/work/dataset/temp/Dataset-PULP-Dronet-V3
python training.py --gpu=0 --block_type=Depthwise --model_name=dw_pw_noby_4.0     --depth_mult=4.0    --arch=dronet_dory_no_residuals --data_path=/home/lamberti/work/dataset/Dataset-PULP-Dronet-V3_aug/  --data_path_testing=/home/lamberti/work/dataset/temp/Dataset-PULP-Dronet-V3
python training.py --gpu=0 --block_type=Depthwise --model_name=dw_pw_noby_8.0     --depth_mult=8.0    --arch=dronet_dory_no_residuals --data_path=/home/lamberti/work/dataset/Dataset-PULP-Dronet-V3_aug/  --data_path_testing=/home/lamberti/work/dataset/temp/Dataset-PULP-Dronet-V3

# # DW+PW with with bypass
python training.py --gpu=0 --block_type=Depthwise --model_name=dw_pw_0.125 --depth_mult=0.125  --arch=dronet_dory --data_path=/home/lamberti/work/dataset/Dataset-PULP-Dronet-V3_aug/  --data_path_testing=/home/lamberti/work/dataset/temp/Dataset-PULP-Dronet-V3
python training.py --gpu=0 --block_type=Depthwise --model_name=dw_pw_0.25  --depth_mult=0.25   --arch=dronet_dory --data_path=/home/lamberti/work/dataset/Dataset-PULP-Dronet-V3_aug/  --data_path_testing=/home/lamberti/work/dataset/temp/Dataset-PULP-Dronet-V3
python training.py --gpu=0 --block_type=Depthwise --model_name=dw_pw_0.5   --depth_mult=0.5    --arch=dronet_dory --data_path=/home/lamberti/work/dataset/Dataset-PULP-Dronet-V3_aug/  --data_path_testing=/home/lamberti/work/dataset/temp/Dataset-PULP-Dronet-V3
python training.py --gpu=0 --block_type=Depthwise --model_name=dw_pw_1.0   --depth_mult=1.0    --arch=dronet_dory --data_path=/home/lamberti/work/dataset/Dataset-PULP-Dronet-V3_aug/  --data_path_testing=/home/lamberti/work/dataset/temp/Dataset-PULP-Dronet-V3
python training.py --gpu=0 --block_type=Depthwise --model_name=dw_pw_2.0   --depth_mult=2.0    --arch=dronet_dory --data_path=/home/lamberti/work/dataset/Dataset-PULP-Dronet-V3_aug/  --data_path_testing=/home/lamberti/work/dataset/temp/Dataset-PULP-Dronet-V3
python training.py --gpu=0 --block_type=Depthwise --model_name=dw_pw_4.0   --depth_mult=4.0    --arch=dronet_dory --data_path=/home/lamberti/work/dataset/Dataset-PULP-Dronet-V3_aug/  --data_path_testing=/home/lamberti/work/dataset/temp/Dataset-PULP-Dronet-V3
python training.py --gpu=0 --block_type=Depthwise --model_name=dw_pw_8.0   --depth_mult=8.0    --arch=dronet_dory --data_path=/home/lamberti/work/dataset/Dataset-PULP-Dronet-V3_aug/  --data_path_testing=/home/lamberti/work/dataset/temp/Dataset-PULP-Dronet-V3

## TESTING
python testing.py --gpu=0 --block_type=ResBlock --model_weights=/home/lamberti/work/tii/pulp-dronet-v3/training/ResBlocks/with_byp/resblock_1.0/resblock_1.0_100.pth   --depth_mult=1.0    --arch=dronet_dory  --data_path=/home/lamberti/work/dataset/temp/Dataset-PULP-Dronet-V3