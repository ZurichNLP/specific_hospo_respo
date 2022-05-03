#!/usr/bin/env bash
# -*- coding: utf-8 -*-

base=`dirname "$(readlink -f "$0")"`

inference_dir="../models/"
# header
grep -P '^,' "$inference_dir/human_ref.eval_result.txt"
# ground truth
grep -P '^/srv' "$inference_dir/human_ref.eval_result.txt"
# rule-based
grep -P '^/srv' "$inference_dir/rule_based/eval_result.txt"
# filtered models
for model_dir in baseline filt_tgt_ppl filt_gen_sent filt_freq_distro filt_tfidf filt_rrsts filt_rrtfidf
do
    grep -P '^/srv' "$inference_dir/$model_dir/inference/eval_result.txt"
done
# labeled models
for id in 0 1 2 3 4 5 n
do
    grep -P '^/srv' "$inference_dir/label_tgt_ppl/inference/ppl${id}_eval_result.txt"
done