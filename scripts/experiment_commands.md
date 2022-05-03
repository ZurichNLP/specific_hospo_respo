This document contains the commands used to perform our experiments. 

Given the datasets used and model dependencies, these commands can be copy-pasted as necessary to reproduce certain experiment results.

## Trained models

```
bash run_finetuning.sh 4 \
  filt_freq_distro \
  ../models/ \
  ../models/huggingface/bart-base/ \
  '../data/hotel/filt_freq_distro_0.0_0.883'
```

```
bash run_finetuning.sh 5 \
  filt_gen_sent \
  ../models/ \
  ../models/huggingface/bart-base/ \
  '../data/hotel/filt_generic_sent_avg_0.0_0.7'
```

```
bash run_finetuning.sh 6 \
  filt_tgt_ppl \
  ../models/ \
  ../models/huggingface/bart-base/ \
  '../data/hotel/filt_tgt_ppl_23.5_50'
```

```
bash run_finetuning.sh 6 \
  filt_tfidf \
  ../models/ \
  ../models/huggingface/bart-base \
  '../data/hotel/filt_response_tfidf_1.37_1.8'  
```

```
bash run_finetuning.sh 6 \
  filt_rrsts \
  ../models/ \
  ../models/huggingface/bart-base \
  '../data/hotel/filt_rev_resp_sts_0.51_0.8'
```

```
bash run_finetuning.sh 6 \
  filt_rrtfidf \
  ../models/ \
  ../models/huggingface/bart-base \
  '../data/hotel/filt_rev_resp_tfidf_0.118_0.6'
```


## Inference runs:

```
bash run_inference.sh inference_filt_freq_distro 4

bash run_inference.sh inference_filt_gen_sent 5

bash run_inference.sh inference_filt_tgt_ppl 6

bash run_inference.sh inference_filt_tfidf 3

bash run_inference.sh inference_filt_rrsts 5

bash run_inference.sh inference_filt_rrtfidf 6

```

## Evaluation:

```
bash run_eval.sh eval_hospo_respo_filtering
```


## Generating data from Human Evaluations

```
python generate_data_for_human_evaluation.py \
  -m ../models/baseline/inference/bs5.txt \
  ../models/filt_freq_distro/inference/bs5.txt \
  ../models/filt_gen_sent/inference/bs5.txt \
  ../models/filt_tgt_ppl/inference/bs5.txt \
  ../models/filt_tfidf/inference/bs5.txt \
  -s ../data/hotel/500k/test.review \
  -r ../data/hotel/500k/test.response \
  -o ../data/hotel/500k/model_outputs_for_human_evalutation_v2.jsonl
```

## Randomized variance: fine-tune and test with different random seed

```
# fine-tune
bash run_finetuning.sh 5 \
  filt_tgt_ppl_s985 \
  ../models/ \
  ../models/huggingface/bart-base/ \
  '../data/hotel/filt_tgt_ppl_23.5_50' \
  985

# inference
bash run_inference.sh inference_filt_tgt_ppl_s985 6

# evaluate
python evaluate_line_aligned.py \
  ../models/filt_tgt_ppl_s985/inference/ckpt07/bs5.txt \
  --src_file ../data/hotel/500k/test.review \
  -ref_file ../data/hotel/500k/test.response \
  --rating_ref ../data/hotel/500k/test.rating \
  --compute_sts | tee ../models/filt_tgt_ppl_s985/inference/ckpt07/eval_result.txt

```

```
# fine-tune
bash run_finetuning.sh 4 \
  filt_tgt_ppl_s42 \
  ../models/ \
  ../models/huggingface/bart-base/ \
  '../data/hotel/filt_tgt_ppl_23.5_50' \
  42

# inference
bash run_inference.sh inference_filt_tgt_ppl_s42 6

# evaluate
python evaluate_line_aligned.py \
  ../models/filt_tgt_ppl_s42/inference/ckpt07/bs5.txt \
  --src_file ../data/hotel/500k/test.review \
  -ref_file ../data/hotel/500k/test.response \
  --rating_ref ../data/hotel/500k/test.rating \
  --compute_sts | tee ../models/filt_tgt_ppl_s42/inference/ckpt07/eval_result.txt
  

# fine-tune seed=42

bash run_finetuning.sh 6 \
  filt_freq_distro_s42 \
  ../models/ \
  ../models/huggingface/bart-base/ \
  '../data/hotel/filt_freq_distro_0.0_0.883' \
  42

bash run_finetuning.sh 6 \
  filt_gen_sent_s42 \
  ../models/ \
  ../models/huggingface/bart-base/ \
  '../data/hotel/filt_generic_sent_avg_0.0_0.7' \
  42

bash run_finetuning.sh 6 \
  baseline_s42 \
  ../models/ \
  ../models/huggingface/bart-base/ \
  '../data/hotel/baseline' \
  42

# inference
bash run_inference inference_baseline_s42 4
bash run_inference inference_filt_freq_distro_s42 4
bash run_inference inference_filt_gen_sent_42 4


# fine-tune seed=985

bash run_finetuning.sh 5 \
  filt_freq_distro_s985 \
  ../models/ \
  ../models/huggingface/bart-base/ \
  '../data/hotel/filt_freq_distro_0.0_0.883' \
  985

bash run_finetuning.sh 5 \
  filt_gen_sent_s985 \
  ../models/ \
  ../models/huggingface/bart-base/ \
  '../data/hotel/filt_generic_sent_avg_0.0_0.7' \
  985


# inference
bash run_inference.sh inference_baseline_s985 6
bash run_inference.sh inference_filt_freq_distro_s985 4
bash run_inference.sh inference_filt_gen_sent_s985 6
```

## Labeled model experiment

```
bash run_finetuning.sh 6 \
  label_tgt_ppl \
  ../models/ \
  ../models/huggingface/bart-base \
  ../data/hotel/labeled_tgt_ppl

bash run_inference_eval_ppl_labels.sh
```


## Ablation Studies

20% train data with LM PPL filtering

```
bash run_finetuning.sh 4 \
  filt_tgt_ppl_abl_20 \
  ../models/ \
  ../models/huggingface/bart-base/ \
  '../data/hotel/filt_tgt_ppl_31.5_50'

bash run_inference.sh inference_filt_tgt_ppl_20 3
```


60% train data with LM PPL filtering

```
bash run_finetuning.sh 5 \
  filt_tgt_ppl_abl_60 \
  ../models/ \
  ../models/huggingface/bart-base/ \
  '../data/hotel/filt_tgt_ppl_18_50'

bash run_inference.sh inference_filt_tgt_ppl_60 5
```

80% train data with LM PPL filtering

```
bash run_finetuning.sh 6 \
  filt_tgt_ppl_abl_80 \
  ../models/ \
  ../models/huggingface/bart-base/ \
  '../data/hotel/filt_tgt_ppl_12_50'

bash run_inference.sh inference_filt_tgt_ppl_80 6
```

combining all filtering methods

```
bash run_finetuning.sh 3 \
  filt_combo \
  ../models/ \
  ../models/huggingface/bart-base/ \
  '../data/hotel/filt_combo'

bash run_inference.sh inference_filt_combo 4
```

## Applying to App review response data (Gao et al. 2019)

To prepare the data, train the scoring LM and generate filtered splits, see `prep_app_data_for_filtering_experiments.ipynb`

```
bash run_finetuning.sh 2 \
  baseline \
  ../models/apps/ \
  ../models/huggingface/bart-base/ \
  '../data/apps/src-tgt/baseline'

bash run_inference.sh inference_app_baseline 6

bash run_finetuning.sh 3 \
  filt_tgt_ppl_3.0_200 \
  ../models/apps/ \
  ../models/huggingface/bart-base/ \
  '../data/apps/src-tgt/filt_tgt_ppl_3.0_200'

bash run_inference.sh inference_app_80 5

bash run_finetuning.sh 4 \
  filt_tgt_ppl_8.5_200 \
  ../models/apps/ \
  ../models/huggingface/bart-base/ \
  '../data/apps/src-tgt/filt_tgt_ppl_8.5_200'

bash run_inference.sh inference_app_60 4

bash run_finetuning.sh 5 \
  filt_tgt_ppl_13.5_200 \
  ../models/apps/ \
  ../models/huggingface/bart-base/ \
  '../data/apps/src-tgt/filt_tgt_ppl_13.5_200'

bash run_inference.sh inference_app_40 3

bash run_finetuning.sh 6 \
  filt_tgt_ppl_23_200 \
  ../models/apps/ \
  ../models/huggingface/bart-base/ \
  '../data/apps/src-tgt/filt_tgt_ppl_23_200'

bash run_inference.sh inference_app_20 2
```