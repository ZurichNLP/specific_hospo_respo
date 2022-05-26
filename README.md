# Hospitality Response Generation

## Data Filtering for more Specific Responses

This repo contains the code for the paper "Improving Specificity in Review Response Generation with Data-Driven Data Filtering".

This work investigates the effect of filtering generic responses from the training data and shows that training on smaller, refined datasets improves specificity of the generated responses.


## Setup

```
# either create from the yml provided
conda env create -f environment.yml 

# or setup manually with requirements.txt
conda create -n hospo_respo_bart python=3.8
conda activate hospo_respo_bart
pip install requirements.txt
```

#### Assumptions for Models, Paths and Directory Structure

The script commands specified make the following assumptions about the directory structure and contents. Note: to reproduce, some paths may need to be adapted!

1. models and data are accessible from the top level. This can easily be achieved by creating sym-links, e.g.
  ```
  ln -s /path/to/storage/data_dir data
  ln -s /path/to/storage/model_dir models
  ```
2. bart-base model has been downloaded from huggingface and is stored in `models/huggingface/bart-base`.
3. if fasttext classifiers are trained for evaluation purposes, these should be stored in `models/classifiers` 


## Data Collection

We provide a script to collect all 500,000 review-response pairs from TripAdvisor used for our study. To reproduce the dataset splits, use

```
python scripts/data_prep/scrape_review_response_pairs.py dataset/train.trip_url data/train.csv
python scripts/data_prep/scrape_review_response_pairs.py dataset/test.trip_url data/test.csv
python scripts/data_prep/scrape_review_response_pairs.py dataset/valid.trip_url data/valid.csv
```

For mobile app data, access to the original dataset needs to be requested from the orignal authors. See https://ieeexplore.ieee.org/document/8952476.

## Scoring Genericness and Filtering

Majority of the scripts used are in the form of Jupyter Notebooks. These are useful for streamlining the data analysis.

The three scoring methods described in the paper are defined in their resepctive IPython Notebooks

```
scripts/data_prep/score_lex_freq.ipynb # word-level
scripts/data_prep/score_sent_avg.ipynb # sentence-level
scripts/data_prep/score_lm_ppl.ipynb # document-level
```

Once scored, filtering can be done using the script `scripts/data_prep/filter_data.ipynb`. This allows for inspecting and filtering data according to score thresholds.

## Model Training, Inference and Evaluating Model Outputs

We fine-tuned our models with pytorch lightning. 
The scipts for model training and inference are in `scripts/modelling/`.

Note: these are adapted from skeleton code used in LongMBART (A. Rios, UZH).

`scripts/modelling/train.py`
  - accepts path to pretrained model dir (from 🤗 Transformers)
  - training, validation, test data
  - runs fine-tuning

`scripts/modelling/inference.py`
  - decodes test set

Scripts for evaluating model outputs are in `scripts/evaluation/`. In the paper, we report a collection of automatic metrics to measure the impact of our data filtering on model outputs.
These metrics are computed with the script `evaluate_line_aligned.py`. For example,

`scripts/evaluation/evaluate_line_aligned.py`
  - accepts line-aligned model outputs, source text and reference text files 
  
### Experimental Pipelines

All commands used to perform our experiments are provided in the bash scripts located in `scripts/`.

For more details on the commands used (e.g. in case of reproduction), see also `scripts/experiment_commands.md`.

#### Finetuning

To finetune a model with default settings, use `run_finetuning.sh`, e.g.

```
bash run_finetuning.sh \
  4 \ # GPU device ID
  filt_freq_distro \  # save directory name (gets appended to out_dir for full path)
  models/ \ # out_dir 
  models/huggingface/bart-base/ \ # model_dir containing original pre-trained model downloaded from HuggingFace
  data/hotel/filt_freq_distro_0.0_0.883 # data_dir containing source target line-aligned files
```

#### Inference

To run inference with a fine-tuned model, use one of the functions in `run_inference.sh`, specifying the GPU device ID, e.g.

```
bash run_inference.sh inference_filt_freq_distro 4
```

#### Automatic Evaluation

To evaluate the model generations, use one of the functions in `run_eval.sh`, e.g.

```
bash run_eval.sh eval_hospo_respo_filtering
```

## Citation

If you find any of these scripts useful for your own research, please cite [Improving Specificity in Review Response Generation with Data-Driven Data Filtering](https://aclanthology.org/2022.ecnlp-1.15) (Kew & Volk, ECNLP 2022).

```
@inproceedings{kew-volk-2022-improving,
    title = "Improving Specificity in Review Response Generation with Data-Driven Data Filtering",
    author = "Kew, Tannon  and
      Volk, Martin",
    booktitle = "Proceedings of The Fifth Workshop on e-Commerce and NLP (ECNLP 5)",
    month = may,
    year = "2022",
    address = "Dublin, Ireland",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2022.ecnlp-1.15",
    pages = "121--133",
}
```
