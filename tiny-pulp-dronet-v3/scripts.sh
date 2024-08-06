#!/bin/sh
# ResBlock with bypass
python training.py --gpu=0 --model_name=resblock_1.0    --block_type=ResBlock   --depth_mult=1.0    --bypass=True
python training.py --gpu=0 --model_name=resblock_0.5    --block_type=ResBlock   --depth_mult=0.5    --bypass=True
python training.py --gpu=0 --model_name=resblock_0.25   --block_type=ResBlock   --depth_mult=0.25   --bypass=True
python training.py --gpu=0 --model_name=resblock_0.125  --block_type=ResBlock   --depth_mult=0.125  --bypass=True

# ResBlock without bypass
python training.py --gpu=0 --model_name=resblock_noby_1.0     --block_type=ResBlock   --depth_mult=1.0    --bypass=False
python training.py --gpu=0 --model_name=resblock_noby_0.5     --block_type=ResBlock   --depth_mult=0.5    --bypass=False
python training.py --gpu=0 --model_name=resblock_noby_0.25    --block_type=ResBlock   --depth_mult=0.25   --bypass=False
python training.py --gpu=0 --model_name=resblock_noby_0.125   --block_type=ResBlock   --depth_mult=0.125  --bypass=False

# DW+PW with with bypass
python training.py --gpu=0 --model_name=dw_pw_1.0       --block_type=Depthwise   --depth_mult=1.0    --bypass=True
python training.py --gpu=0 --model_name=dw_pw_0.5       --block_type=Depthwise   --depth_mult=0.5    --bypass=True
python training.py --gpu=0 --model_name=dw_pw_0.25      --block_type=Depthwise   --depth_mult=0.25   --bypass=True
python training.py --gpu=0 --model_name=dw_pw_0.125     --block_type=Depthwise   --depth_mult=0.125  --bypass=True

# DW+PW without bypass
python training.py --gpu=0 --model_name=dw_pw_noby_1.0   --block_type=Depthwise  --depth_mult=1.0    --bypass=False
python training.py --gpu=0 --model_name=dw_pw_noby_0.5   --block_type=Depthwise  --depth_mult=0.5    --bypass=False
python training.py --gpu=0 --model_name=dw_pw_noby_0.25  --block_type=Depthwise  --depth_mult=0.25   --bypass=False
python training.py --gpu=0 --model_name=dw_pw_noby_0.125 --block_type=Depthwise  --depth_mult=0.125  --bypass=False

# IRLB with bypass
python training.py --gpu=0 --model_name=irlb_1.0       --block_type=IRLB   --depth_mult=1.0    --bypass=True
python training.py --gpu=0 --model_name=irlb_0.5       --block_type=IRLB   --depth_mult=0.5    --bypass=True
python training.py --gpu=0 --model_name=irlb_0.25      --block_type=IRLB   --depth_mult=0.25   --bypass=True
python training.py --gpu=0 --model_name=irlb_0.125     --block_type=IRLB   --depth_mult=0.125  --bypass=True

# IRLB without bypass
python training.py --gpu=0 --model_name=irlb_noby_1.0   --block_type=IRLB  --depth_mult=1.0    --bypass=False
python training.py --gpu=0 --model_name=irlb_noby_0.5   --block_type=IRLB  --depth_mult=0.5    --bypass=False
python training.py --gpu=0 --model_name=irlb_noby_0.25  --block_type=IRLB  --depth_mult=0.25   --bypass=False
python training.py --gpu=0 --model_name=irlb_noby_0.125 --block_type=IRLB  --depth_mult=0.125  --bypass=False

## TESTING
python testing.py --gpu=0 --block_type=ResBlock --model_weights./training/ResBlocks/with_byp/resblock_1.0/resblock_1.0_100.pth   --depth_mult=1.0    --bypass=True