# !/bin/bash

seq_num=0
source your_env_path/anaconda3/bin/activate
conda activate your_env
export CUDA_VISIBLE_DEVICES=0  # your GPU number
for file in `<./dataloaders/YouTube_Object_test.txt`
do
    let seq_num++
    echo 'sequence NO.'"${seq_num}"': '"${file}"
    export SEQ_NAME=$file
    python ./infer.py --dset youtube_object --results-dir results_ytboj_multi --multi
done