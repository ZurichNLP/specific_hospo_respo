#!/usr/bin/env bash
# -*- coding: utf-8 -*-

base=`dirname "$(readlink -f "$0")"`
eval_scripts="$base/evaluation"

# hotel data
raw_data="../data/hotel/500k/"
inference_dir="../models/"

####################################################
# ablation studies: Does it generalise to app rrgen?
####################################################

eval_app_data() {

    raw_data="../data/apps/src-tgt/baseline/"
    inference_dir="../models/apps/"

    # evaluating performance on app data
    for model_dir in baseline filt_tgt_ppl_3.0_200 filt_tgt_ppl_8.5_200 filt_tgt_ppl_13.5_200 filt_tgt_ppl_23_200
    do
        translations="$inference_dir/$model_dir/inference/bs5.txt"
        echo "evaluating $translations ..."
        python $eval_scripts/evaluate_line_aligned.py \
            $translations \
            --src_file $raw_data/test.review \
            --ref_file $raw_data/test.response \
            --compute_sts | tee "$inference_dir/$model_dir/inference/eval_result.txt"
    done

}

###################################################
# ablation studies: How much data should we filter?
###################################################

eval_hospo_ablation() {

    for model_dir in filt_tgt_ppl_abl_20 filt_tgt_ppl_abl_60 filt_tgt_ppl_abl_80 filt_combo
    do
        translations="$inference_dir/$model_dir/inference/bs5.txt"
        echo "evaluating $translations ..."
        python $eval_scripts/evaluate_line_aligned.py \
            $translations \
            --src_file $raw_data/test.review \
            --ref_file $raw_data/test.response \
            --compute_sts | tee "$inference_dir/$model_dir/inference/eval_result.txt"
    done

}

eval_hospo_seed_runs() {

    for model_dir in baseline_s42 baseline_s985 filt_gen_sent_s42 filt_gen_sent_s985 filt_freq_distro_s42 filt_freq_distro_s985
    do
        translations="$inference_dir/$model_dir/inference/bs5.txt"
        echo "evaluating $translations ..."
        python $eval_scripts/evaluate_line_aligned.py \
            $translations \
            --src_file $raw_data/test.review \
            --ref_file $raw_data/test.response \
            --compute_sts | tee "$inference_dir/$model_dir/inference/eval_result.txt"
    done

}

eval_hospo_respo_filtering() {

    for model_dir in baseline filt_tgt_ppl filt_gen_sent filt_freq_distro
    do
        translations="$inference_dir/$model_dir/inference/bs5.txt"
        echo "evaluating $translations ..."
        python $eval_scripts/evaluate_line_aligned.py \
            $translations \
            --src_file $raw_data/test.review \
            --ref_file $raw_data/test.response \
            --compute_sts | tee "$inference_dir/$model_dir/inference/eval_result.txt"
    done

}

eval_rulebased_baseline() {

    # rule based lookup
    echo "evaluating $inference_dir/rule_based/translations.txt ..."
    python $eval_scripts/evaluate_line_aligned.py \
        $inference_dir/rule_based/translations.txt \
        --src_file $raw_data/test.review \
        --ref_file $raw_data/test.response \
        --compute_sts | tee "$inference_dir/rule_based/eval_result.txt"

}

eval_human_refs() {

    # human references
    echo "$raw_data/test.response"
    python $eval_scripts/evaluate_line_aligned.py \
        $raw_data/test.response \
        --src_file $raw_data/test.review \
        --ref_file $raw_data/test.response \
        --compute_sts | tee "$inference_dir/human_ref.eval_result.txt"

}

#######################################################
# ablation studies: Should we filter or label the data?
#######################################################

eval_labelled_generation() {
 
    for id in 0 1 2 3 4 5 n
    do
        translations="$inference_dir/label_tgt_ppl/inference/ppl${id}_bs5.txt"
        echo "evaluating $translations ..."
        python $eval_scripts/evaluate_line_aligned.py \
            $translations \
            --src_file $raw_data/test.review \
            --ref_file $raw_data/test.response \
            --compute_sts | tee "$inference_dir/label_tgt_ppl/inference/ppl${id}_eval_result.txt"
        echo $id
    done

}


"$@"
