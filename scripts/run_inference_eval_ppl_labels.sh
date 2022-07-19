#!/usr/bin/env bash
# -*- coding: utf-8 -*-

set -e


base=`dirname "$(readlink -f "$0")"`

model_dir=../models/label_tgt_ppl
data_dir=../data/hotel/500k/
out_dir=$model_dir/inference/
eval_scripts=evaluations/
scripts=modelling/

GPU=6
export CUDA_VISIBLE_DEVICES=$GPU 

for n in 0 1 2 3 4 5 n
do
    outfile="$out_dir/ppl${n}_bs5.txt"
    echo "setting ppl label = $n and decoding ..."
    echo "writing outputs to $outfile ..."
    # use the customised test set from with "ppl_3:" labels
    python $scripts/inference.py \
        --model_path $model_dir \
        --checkpoint_name 'checkpointepoch=07_rouge2=0.15611.ckpt' \
        --test_source $data_dir/test.ppl${n}pref_review \
        --test_target $data_dir/test.response \
        --max_input_len 512 --max_output_len 512 \
        --num_workers 8 \
        --progress_bar_refresh_rate 1 \
        --batch_size 5 \
        --beam_size 5 --num_return_sequences 1 \
        --translation "$outfile" | tee "$out_dir/ppl${n}_bs5.decode.log"
        
    echo "evaluating $outfile ..."
    python $eval_scripts/evaluate_line_aligned.py \
        --hyp_files $outfile \
        --src_file $data_dir/test.review \
        --ref_file $data_dir/test.response \
        --compute_sts | tee "$out_dir/ppl${n}_bs5_eval_result.txt"
done

echo "done!"