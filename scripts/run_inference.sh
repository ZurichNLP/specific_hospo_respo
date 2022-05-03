#!/usr/bin/env bash
# -*- coding: utf-8 -*-

scripts="./modelling"
data_dir="../data/hotel/500k"

inference_baseline() {
    GPU=$1
    model_dir=../models/baseline/
    out_dir=$model_dir/inference/
    mkdir -p $out_dir

    echo "running inference on GPU $GPU"
    CUDA_VISIBLE_DEVICES=$GPU nohup python $scripts/inference.py \
        --model_path $model_dir \
        --checkpoint_name 'checkpointepoch=05_rouge2=0.16466.ckpt' \
        --test_source $data_dir/test.review \
        --test_target $data_dir/test.response \
        --max_input_len 512 --max_output_len 512 \
        --num_workers 8 \
        --progress_bar_refresh_rate 1 \
        --batch_size 5 \
        --beam_size 5 --num_return_sequences 1 \
        --translation "$out_dir/bs5.txt" >| "$out_dir/bs5.decode.log" &
}

inference_baseline_s42() {
    GPU=$1
    model_dir=../models/baseline_s42/
    out_dir=$model_dir/inference/
    mkdir -p $out_dir

    echo "running inference on GPU $GPU"
    CUDA_VISIBLE_DEVICES=$GPU nohup python $scripts/inference.py \
        --model_path $model_dir \
        --checkpoint_name 'checkpointepoch=05_rouge2=0.16906.ckpt' \
        --test_source $data_dir/test.review \
        --test_target $data_dir/test.response \
        --max_input_len 512 --max_output_len 512 \
        --num_workers 8 \
        --progress_bar_refresh_rate 1 \
        --batch_size 5 \
        --beam_size 5 --num_return_sequences 1 \
        --translation "$out_dir/bs5.txt" >| "$out_dir/bs5.decode.log" &
}

inference_baseline_s985() {
    GPU=$1
    model_dir=../models/baseline_s985/
    out_dir=$model_dir/inference/
    mkdir -p $out_dir

    echo "running inference on GPU $GPU"
    CUDA_VISIBLE_DEVICES=$GPU nohup python $scripts/inference.py \
        --model_path $model_dir \
        --checkpoint_name 'checkpointepoch=05_rouge2=0.17016.ckpt' \
        --test_source $data_dir/test.review \
        --test_target $data_dir/test.response \
        --max_input_len 512 --max_output_len 512 \
        --num_workers 8 \
        --progress_bar_refresh_rate 1 \
        --batch_size 5 \
        --beam_size 5 --num_return_sequences 1 \
        --translation "$out_dir/bs5.txt" >| "$out_dir/bs5.decode.log" &
}


inference_filt_freq_distro() {
    GPU=$1
    model_dir=../models/filt_freq_distro/
    out_dir=$model_dir/inference/
    mkdir -p $out_dir

    echo "running inference on GPU $GPU"
    CUDA_VISIBLE_DEVICES=$GPU nohup python $scripts/inference.py \
        --model_path $model_dir \
        --checkpoint_name 'checkpointepoch=07_rouge2=0.14441.ckpt' \
        --test_source $data_dir/test.review \
        --test_target $data_dir/test.response \
        --max_input_len 512 --max_output_len 512 \
        --num_workers 8 \
        --progress_bar_refresh_rate 1 \
        --batch_size 24 \
        --beam_size 5 --num_return_sequences 1 \
        --translation "$out_dir/bs5.txt" >| "$out_dir/bs5.decode.log" &
}


inference_filt_freq_distro_s42() {
    GPU=$1
    model_dir=../models/filt_freq_distro_s42/
    out_dir=$model_dir/inference/
    mkdir -p $out_dir

    echo "running inference on GPU $GPU"
    CUDA_VISIBLE_DEVICES=$GPU nohup python $scripts/inference.py \
        --model_path $model_dir \
        --checkpoint_name 'checkpointepoch=06_rouge2=0.15364.ckpt' \
        --test_source $data_dir/test.review \
        --max_input_len 512 --max_output_len 512 \
        --num_workers 8 \
        --progress_bar_refresh_rate 1 \
        --batch_size 5 \
        --beam_size 5 --num_return_sequences 1 \
        --translation "$out_dir/bs5.txt" >| "$out_dir/bs5.decode.log" &
}


inference_filt_freq_distro_s985() {
    GPU=$1
    model_dir=../models/filt_freq_distro_s985/
    out_dir=$model_dir/inference/
    mkdir -p $out_dir

    echo "running inference on GPU $GPU"
    CUDA_VISIBLE_DEVICES=$GPU nohup python $scripts/inference.py \
        --model_path $model_dir \
        --checkpoint_name 'checkpointepoch=07_rouge2=0.14722.ckpt' \
        --test_source $data_dir/test.review \
        --max_input_len 512 --max_output_len 512 \
        --num_workers 8 \
        --progress_bar_refresh_rate 1 \
        --batch_size 5 \
        --beam_size 5 --num_return_sequences 1 \
        --translation "$out_dir/bs5.txt" >| "$out_dir/bs5.decode.log" &
}

inference_filt_gen_sent() {
    GPU=$1
    model_dir=../models/filt_gen_sent/
    out_dir=$model_dir/inference/
    mkdir -p $out_dir

    CUDA_VISIBLE_DEVICES=$GPU nohup python $scripts/inference.py \
        --model_path $model_dir \
        --checkpoint_name 'checkpointepoch=07_rouge2=0.12060.ckpt' \
        --test_source $data_dir/test.review \
        --test_target $data_dir/test.response \
        --max_input_len 512 --max_output_len 512 \
        --num_workers 8 \
        --progress_bar_refresh_rate 1 \
        --batch_size 10 \
        --beam_size 5 --num_return_sequences 1 \
        --translation "$out_dir/bs5.txt" >| "$out_dir/bs5.decode.log" &
}

inference_filt_gen_sent_s42() {
    GPU=$1
    model_dir=../models/filt_gen_sent_s42/
    out_dir=$model_dir/inference/
    mkdir -p $out_dir

    CUDA_VISIBLE_DEVICES=$GPU nohup python $scripts/inference.py \
        --model_path $model_dir \
        --checkpoint_name 'checkpointepoch=07_rouge2=0.12164.ckpt' \
        --test_source $data_dir/test.review \
        --max_input_len 512 --max_output_len 512 \
        --num_workers 8 \
        --progress_bar_refresh_rate 1 \
        --batch_size 5 \
        --beam_size 5 --num_return_sequences 1 \
        --translation "$out_dir/bs5.txt" >| "$out_dir/bs5.decode.log" &
}

inference_filt_gen_sent_s985() {
    GPU=$1
    model_dir=../models/filt_gen_sent_s985/
    out_dir=$model_dir/inference/
    mkdir -p $out_dir

    CUDA_VISIBLE_DEVICES=$GPU nohup python $scripts/inference.py \
        --model_path $model_dir \
        --checkpoint_name 'checkpointepoch=07_rouge2=0.11884.ckpt' \
        --test_source $data_dir/test.review \
        --max_input_len 512 --max_output_len 512 \
        --num_workers 8 \
        --progress_bar_refresh_rate 1 \
        --batch_size 5 \
        --beam_size 5 --num_return_sequences 1 \
        --translation "$out_dir/bs5.txt" >| "$out_dir/bs5.decode.log" &
}


inference_filt_tgt_ppl() {
    GPU=$1
    model_dir=../models/filt_tgt_ppl/
    out_dir=$model_dir/inference/
    mkdir -p $out_dir

    CUDA_VISIBLE_DEVICES=$GPU nohup python $scripts/inference.py \
        --model_path $model_dir \
        --checkpoint_name 'checkpointepoch=07_rouge2=0.13102.ckpt' \
        --test_source $data_dir/test.review \
        --test_target $data_dir/test.response \
        --max_input_len 512 --max_output_len 512 \
        --num_workers 8 \
        --progress_bar_refresh_rate 1 \
        --batch_size 24 \
        --beam_size 5 --num_return_sequences 1 \
        --translation "$out_dir/bs5.txt" >| "$out_dir/bs5.decode.log" &
}

inference_filt_tfidf() {
    GPU=$1
    model_dir=../models/filt_tfidf/
    out_dir=$model_dir/inference/
    mkdir -p $out_dir

    CUDA_VISIBLE_DEVICES=$GPU nohup python $scripts/inference.py \
        --model_path $model_dir \
        --checkpoint_name 'checkpointepoch=07_rouge2=0.14807.ckpt' \
        --test_source $data_dir/test.review \
        --test_target $data_dir/test.response \
        --max_input_len 512 --max_output_len 512 \
        --num_workers 8 \
        --progress_bar_refresh_rate 1 \
        --batch_size 5 \
        --beam_size 5 --num_return_sequences 1 \
        --translation "$out_dir/bs5.txt" >| "$out_dir/bs5.decode.log" &
}


inference_filt_rrsts() {
    GPU=$1
    model_dir=../models/filt_rrsts/
    out_dir=$model_dir/inference/
    mkdir -p $out_dir

    CUDA_VISIBLE_DEVICES=$GPU nohup python $scripts/inference.py \
        --model_path $model_dir \
        --checkpoint_name 'checkpointepoch=07_rouge2=0.17882.ckpt' \
        --test_source $data_dir/test.review \
        --test_target $data_dir/test.response \
        --max_input_len 512 --max_output_len 512 \
        --num_workers 8 \
        --progress_bar_refresh_rate 1 \
        --batch_size 5 \
        --beam_size 5 --num_return_sequences 1 \
        --translation "$out_dir/bs5.txt" >| "$out_dir/bs5.decode.log" &
}


inference_filt_rrtfidf() {
    GPU=$1
    model_dir=../models/filt_rrtfidf/
    out_dir=$model_dir/inference/
    mkdir -p $out_dir

    CUDA_VISIBLE_DEVICES=$GPU nohup python $scripts/inference.py \
        --model_path $model_dir \
        --checkpoint_name 'checkpointepoch=07_rouge2=0.17964.ckpt' \
        --test_source $data_dir/test.review \
        --test_target $data_dir/test.response \
        --max_input_len 512 --max_output_len 512 \
        --num_workers 8 \
        --progress_bar_refresh_rate 1 \
        --batch_size 5 \
        --beam_size 5 --num_return_sequences 1 \
        --translation "$out_dir/bs5.txt" >| "$out_dir/bs5.decode.log" &
}


inference_filt_tgt_ppl_s985() {
    GPU=$1
    model_dir=../models/filt_tgt_ppl_s985
    out_dir=$model_dir/inference/ckpt07
    mkdir -p $out_dir

    CUDA_VISIBLE_DEVICES=$GPU nohup python $scripts/inference.py \
        --model_path $model_dir \
        --checkpoint_name 'checkpointepoch=07_rouge2=0.13078.ckpt' \
        --test_source $data_dir/test.review \
        --test_target $data_dir/test.response \
        --max_input_len 512 --max_output_len 512 \
        --num_workers 8 \
        --progress_bar_refresh_rate 1 \
        --batch_size 5 \
        --beam_size 5 --num_return_sequences 1 \
        --translation "$out_dir/bs5.txt" >| "$out_dir/bs5.decode.log" &
}

inference_filt_tgt_ppl_s42() {
    GPU=$1
    model_dir=../models/filt_tgt_ppl_s42
    out_dir=$model_dir/inference/ckpt07
    mkdir -p $out_dir

    CUDA_VISIBLE_DEVICES=$GPU nohup python $scripts/inference.py \
        --model_path $model_dir \
        --checkpoint_name 'checkpointepoch=07_rouge2=0.12659.ckpt' \
        --test_source $data_dir/test.review \
        --test_target $data_dir/test.response \
        --max_input_len 512 --max_output_len 512 \
        --num_workers 8 \
        --progress_bar_refresh_rate 1 \
        --batch_size 5 \
        --beam_size 5 --num_return_sequences 1 \
        --translation "$out_dir/bs5.txt" >| "$out_dir/bs5.decode.log" &
}



inference_filt_tgt_ppl_20() {
    GPU=$1
    model_dir=../models/filt_tgt_ppl_abl_20
    out_dir=$model_dir/inference/
    mkdir -p $out_dir

    CUDA_VISIBLE_DEVICES=$GPU nohup python $scripts/inference.py \
        --model_path $model_dir \
        --checkpoint_name 'checkpointepoch=00_rouge2=0.11408.ckpt' \
        --test_source $data_dir/test.review \
        --test_target $data_dir/test.response \
        --max_input_len 512 --max_output_len 512 \
        --num_workers 8 \
        --progress_bar_refresh_rate 1 \
        --batch_size 5 \
        --beam_size 5 --num_return_sequences 1 \
        --translation "$out_dir/bs5.txt" >| "$out_dir/bs5.decode.log" &
}

inference_filt_tgt_ppl_60() {
    GPU=$1
    model_dir=../models/filt_tgt_ppl_abl_60
    out_dir=$model_dir/inference/
    mkdir -p $out_dir

    CUDA_VISIBLE_DEVICES=$GPU nohup python $scripts/inference.py \
        --model_path $model_dir \
        --checkpoint_name 'checkpointepoch=06_rouge2=0.15659.ckpt' \
        --test_source $data_dir/test.review \
        --test_target $data_dir/test.response \
        --max_input_len 512 --max_output_len 512 \
        --num_workers 8 \
        --progress_bar_refresh_rate 1 \
        --batch_size 5 \
        --beam_size 5 --num_return_sequences 1 \
        --translation "$out_dir/bs5.txt" >| "$out_dir/bs5.decode.log" &
}

inference_filt_tgt_ppl_80() {
    GPU=$1
    model_dir=../models/filt_tgt_ppl_abl_80
    out_dir=$model_dir/inference/
    mkdir -p $out_dir

    CUDA_VISIBLE_DEVICES=$GPU nohup python $scripts/inference.py \
        --model_path $model_dir \
        --checkpoint_name 'checkpointepoch=03_rouge2=0.17286.ckpt' \
        --test_source $data_dir/test.review \
        --test_target $data_dir/test.response \
        --max_input_len 512 --max_output_len 512 \
        --num_workers 8 \
        --progress_bar_refresh_rate 1 \
        --batch_size 5 \
        --beam_size 5 --num_return_sequences 1 \
        --translation "$out_dir/bs5.txt" >| "$out_dir/bs5.decode.log" &
}


inference_filt_combo() {
    GPU=$1
    model_dir=../models/filt_combo
    out_dir=$model_dir/inference/
    mkdir -p $out_dir

    CUDA_VISIBLE_DEVICES=$GPU nohup python $scripts/inference.py \
        --model_path $model_dir \
        --checkpoint_name 'checkpointepoch=07_rouge2=0.10821.ckpt' \
        --test_source $data_dir/test.review \
        --test_target $data_dir/test.response \
        --max_input_len 512 --max_output_len 512 \
        --num_workers 8 \
        --progress_bar_refresh_rate 1 \
        --batch_size 5 \
        --beam_size 5 --num_return_sequences 1 \
        --translation "$out_dir/bs5.txt" >| "$out_dir/bs5.decode.log" &
}


inference_label_tgt_ppl_n() {
    GPU=$1
    n=${2:-3} # default=3
    model_dir=../models/label_tgt_ppl
    out_dir=$model_dir/inference/
    mkdir -p $out_dir
    # use the customised test set from with "ppl_3:" labels
    CUDA_VISIBLE_DEVICES=$GPU nohup python $scripts/inference.py \
        --model_path $model_dir \
        --checkpoint_name 'checkpointepoch=07_rouge2=0.15611.ckpt' \
        --test_source $data_dir/test.ppl${n}pref_review \
        --test_target $data_dir/test.response \
        --max_input_len 512 --max_output_len 512 \
        --num_workers 8 \
        --progress_bar_refresh_rate 1 \
        --batch_size 10 \
        --beam_size 5 --num_return_sequences 1 \
        --translation "$out_dir/ppl${n}_bs5.txt" >| "$out_dir/ppl${n}_bs5.decode.log" &
}


inference_app_baseline() {
    GPU=$1
    model_dir=../models/apps/baseline
    data_dir=../data/apps/src-tgt/baseline
    out_dir=$model_dir/inference/
    mkdir -p $out_dir

    CUDA_VISIBLE_DEVICES=$GPU nohup python $scripts/inference.py \
        --model_path $model_dir \
        --checkpoint_name 'checkpointepoch=03_rouge2=0.21807.ckpt' \
        --test_source $data_dir/test.review \
        --test_target $data_dir/test.response \
        --max_input_len 512 --max_output_len 512 \
        --num_workers 8 \
        --progress_bar_refresh_rate 1 \
        --batch_size 5 \
        --beam_size 5 --num_return_sequences 1 \
        --translation "$out_dir/bs5.txt" >| "$out_dir/bs5.decode.log" &
}

inference_app_20() {
    GPU=$1
    model_dir=../models/apps/filt_tgt_ppl_23_200
    data_dir=../data/apps/src-tgt/baseline
    out_dir=$model_dir/inference/
    mkdir -p $out_dir

    CUDA_VISIBLE_DEVICES=$GPU nohup python $scripts/inference.py \
        --model_path $model_dir \
        --checkpoint_name 'checkpointepoch=14_rouge2=0.22479.ckpt' \
        --test_source $data_dir/test.review \
        --test_target $data_dir/test.response \
        --max_input_len 512 --max_output_len 512 \
        --num_workers 8 \
        --progress_bar_refresh_rate 1 \
        --batch_size 5 \
        --beam_size 5 --num_return_sequences 1 \
        --translation "$out_dir/bs5.txt" >| "$out_dir/bs5.decode.log" &
}

inference_app_40() {
    GPU=$1
    model_dir=../models/apps/filt_tgt_ppl_13.5_200
    data_dir=../data/apps/src-tgt/baseline
    out_dir=$model_dir/inference/
    mkdir -p $out_dir

    CUDA_VISIBLE_DEVICES=$GPU nohup python $scripts/inference.py \
        --model_path $model_dir \
        --checkpoint_name 'checkpointepoch=11_rouge2=0.17716.ckpt' \
        --test_source $data_dir/test.review \
        --test_target $data_dir/test.response \
        --max_input_len 512 --max_output_len 512 \
        --num_workers 8 \
        --progress_bar_refresh_rate 1 \
        --batch_size 5 \
        --beam_size 5 --num_return_sequences 1 \
        --translation "$out_dir/bs5.txt" >| "$out_dir/bs5.decode.log" &
}


inference_app_60() {
    GPU=$1
    model_dir=../models/apps/filt_tgt_ppl_8.5_200
    data_dir=../data/apps/src-tgt/baseline
    out_dir=$model_dir/inference/
    mkdir -p $out_dir

    CUDA_VISIBLE_DEVICES=$GPU nohup python $scripts/inference.py \
        --model_path $model_dir \
        --checkpoint_name 'checkpointepoch=09_rouge2=0.13567.ckpt' \
        --test_source $data_dir/test.review \
        --test_target $data_dir/test.response \
        --max_input_len 512 --max_output_len 512 \
        --num_workers 8 \
        --progress_bar_refresh_rate 1 \
        --batch_size 5 \
        --beam_size 5 --num_return_sequences 1 \
        --translation "$out_dir/bs5.txt" >| "$out_dir/bs5.decode.log" &
}

inference_app_80() {
    GPU=$1
    model_dir=../models/apps/filt_tgt_ppl_3.0_200
    data_dir=../data/apps/src-tgt/baseline
    out_dir=$model_dir/inference/
    mkdir -p $out_dir

    CUDA_VISIBLE_DEVICES=$GPU nohup python $scripts/inference.py \
        --model_path $model_dir \
        --checkpoint_name 'checkpointepoch=05_rouge2=0.16095.ckpt' \
        --test_source $data_dir/test.review \
        --test_target $data_dir/test.response \
        --max_input_len 512 --max_output_len 512 \
        --num_workers 8 \
        --progress_bar_refresh_rate 1 \
        --batch_size 5 \
        --beam_size 5 --num_return_sequences 1 \
        --translation "$out_dir/bs5.txt" >| "$out_dir/bs5.decode.log" &
}


"$@"
