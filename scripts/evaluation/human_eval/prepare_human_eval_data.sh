#!/usr/bin/env bash
# -*- coding: utf-8 -*-

data_dir="../hotel/500k"
models="../models"

# annotators=4
annotation_sets=2
items_per_anno=200
n=$(($annotation_sets * $items_per_anno))
outfile="./model_outputs_for_human_evalutation_$n.jsonl"

python generate_data_for_human_evaluation.py \
    -m $models/baseline/inference/bs5.txt \
    $models/filt_freq_distro/inference/bs5.txt \
    $models/filt_gen_sent/inference/bs5.txt \
    $models/filt_tgt_ppl/inference/bs5.txt \
    -s $data_dir/test.review \
    -r $data_dir/test.response \
    -n $n \
    -o $outfile

split -d -l $((`wc -l < $outfile`/$annotation_sets)) $outfile --additional-suffix=".jsonl"