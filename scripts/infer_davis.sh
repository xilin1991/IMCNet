# !/bin/bash

seq_num=0
source your_env_path/anaconda3/bin/activate
conda activate your_env
export CUDA_VISIBLE_DEVICES=0  # your GPU number
for file in `<./dataloaders/DAVIS16_test.txt`
do
    let seq_num++
    echo 'sequence NO.'"${seq_num}"': '"${file}"
    export SEQ_NAME=$file
    python ./infer.py --dset davis2016 --results-dir results_davis
done