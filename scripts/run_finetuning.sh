#!/usr/bin/env bash
# -*- coding: utf-8 -*-

GPU=$1
prefix_id=$2
out_dir=$3
model_dir=$4
data_dir=$5
seed=${6:-222}

scripts="../modelling/"

echo "GPU: $GPU"
echo "prefix id: $prefix_id"
echo "output dir: $out_dir"
echo "model dir: $model_dir"
echo "data dir: $data_dir"
echo "seed: $seed"

mkdir -p $out_dir/$prefix_id/

CUDA_VISIBLE_DEVICES=$GPU \
    nohup \
    python $scripts/train.py \
    --from_pretrained $model_dir \
    --save_dir $out_dir --save_prefix $prefix_id \
    --train_source $data_dir/train.review \
    --train_target $data_dir/train.response \
    --val_source $data_dir/valid.review \
    --val_target $data_dir/valid.response \
    --test_source $data_dir/test.review \
    --test_target $data_dir/test.response \
    --max_input_len 512 --max_output_len 512 \
    --seed $seed --num_workers 8 \
    --attention_dropout 0.1 --dropout 0.1 \
    --label_smoothing 0.1 \
    --grad_ckpt \
    --progress_bar_refresh_rate 1 \
    --val_check_interval 1 \
    --val_percent_check 25 \
    --test_percent_check 1.0 \
    --lr 0.00003 \
    --batch_size 10 --grad_accum 4 \
    --gpus 1 \
    --max_epochs 10 --early_stopping_metric 'rouge2' --patience 6 \
    --wandb readvisor >| $out_dir/$prefix_id/train.log &
