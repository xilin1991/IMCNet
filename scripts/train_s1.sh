# !/bin/bash

source your_env_path/anaconda3/bin/activate
conda activate your_envs
export CUDA_VISIBLE_DEVICES= 0  # your GPU number
python train_s1.py