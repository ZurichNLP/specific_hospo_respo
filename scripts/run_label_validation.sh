#!/usr/bin/env bash
# -*- coding: utf-8 -*-

set -e

GPU=$1

base=`dirname "$(readlink -f "$0")"`

data_dir=../data/hotel/labeled_tgt_ppl
model_dir=../models/label_tgt_ppl
scripts=$base/scripts
eval_scripts=$base/eval/
out_dir=$model_dir/valid_inference/

mkdir -p $out_dir
export CUDA_VISIBLE_DEVICES=$GPU

tail -n 5000 $data_dir/valid.response > $data_dir/valid.t5k_response  

for n in 3 4 5
do
    tail -n 5000 $data_dir/valid.review | sed "s/ppl_[0-9]: /ppl_${n}: /" > $data_dir/valid.ppl${n}pref_t5k_review
    
    echo "running inference on validation set with ppl_${n} ..."
    python $scripts/inference.py \
        --model_path $model_dir \
        --checkpoint_name 'checkpointepoch=07_rouge2=0.15611.ckpt' \
        --test_source $data_dir/valid.ppl${n}pref_t5k_review \
        --test_target $data_dir/valid.t5k_response \
        --max_input_len 512 --max_output_len 512 \
        --num_workers 8 \
        --progress_bar_refresh_rate 1 \
        --batch_size 5 \
        --beam_size 5 --num_return_sequences 1 \
        --translation $out_dir/ppl${n}_bs5.txt | tee $out_dir/ppl${n}_bs5.decode.log
    
    exit_status=$?
    if [ "${exit_status}" -ne 0 ];
    then
        echo "failed to finish inference ${exit_status}"
        break
    fi

    echo "evaluating ..."

    python $eval_scripts/evaluate_line_aligned.py \
        $out_dir/ppl${n}_bs5.txt \
        --src_file $data_dir/valid.ppl${n}pref_t5k_review \
        --ref_file $data_dir/valid.t5k_response | tee $out_dir/ppl${n}_bs5_eval_result.txt
    
    echo "finished validating ppl_${n} ..."

done

# prep_data=1

# tail 5k counts | all counts
# 1137 ppl_0 |5586
# 1413 ppl_1 | 7068
# 958 ppl_2 | 4923
# 578 ppl_3 | 2794
# 328 ppl_4 | 1655
# 586 ppl_5 | 2871

# ppl_1:
# if [[ $prep_data != 0 ]]; then
#     echo "generating data ..."
#     sed 's/ppl_[0-9]: /ppl_0: /' $data_dir/valid.review | tail -n 5000 > $data_dir/valid.ppl0pref_t5k_review
#     sed 's/ppl_[0-9]: /ppl_1: /' $data_dir/valid.review | tail -n 5000 > $data_dir/valid.ppl1pref_t5k_review
#     sed 's/ppl_[0-9]: /ppl_2: /' $data_dir/valid.review | tail -n 5000 > $data_dir/valid.ppl2pref_t5k_review
#     sed 's/ppl_[0-9]: /ppl_3: /' $data_dir/valid.review | tail -n 5000 > $data_dir/valid.ppl3pref_t5k_review
#     sed 's/ppl_[0-9]: /ppl_4: /' $data_dir/valid.review | tail -n 5000 > $data_dir/valid.ppl4pref_t5k_review
#     sed 's/ppl_[0-9]: /ppl_5: /' $data_dir/valid.review | tail -n 5000 > $data_dir/valid.ppl5pref_t5k_review
# fi
