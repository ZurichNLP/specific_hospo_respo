import json
import argparse
from pathlib import Path
from collections import defaultdict

from tqdm import tqdm
from typing import List, Dict, Tuple
import pandas as pd

import random

SEED = 42
random.seed(SEED)

"""
UPDATED: 16.07.21

Example call:
    python generate_data_for_human_evaluation.py \
        -m ../models/baseline/inference/bs5.txt \
        ../models/filt_freq_distro/inference/bs5.txt \
        ../models/filt_gen_sent/inference/bs5.txt \
        ../models/filt_tgt_ppl/inference/bs5.txt \
        -s ../data/hotel/500k/test.review \
        -r ../data/hotel/500k/test.response
"""

def set_args():
    ap = argparse.ArgumentParser()
    ap.add_argument('-m', '--hypothesis_texts', nargs='*', required=True, help='file(s) containing line-aligned model outputs')
    ap.add_argument('-r', '--reference_texts', required=False, help='file(s) containing line-aligned references')
    ap.add_argument('-s', '--source_texts', required=False, help='file(s) containing line-aligned source texts')
    ap.add_argument('-o', '--outfile', required=True, help='filepath to output JSONL/csv file that is read in by Prodigy/Turkle')
    ap.add_argument('-n', type=int, default=50, help='number of items to output.')
    return ap.parse_args()

def write_to_jsonl_format(df, outfile):
    """
    Writes a simple JSONL format with each row being a
    dictionary containing:
        - source text
        - reference text
        - testset_id
        - model text(s)
    """
    with open(outfile, 'w', encoding='utf8') as outf:    
        for k, row in df.iterrows():
            outf.write(f'{json.dumps(row.to_dict(), ensure_ascii=False,)}\n')
    return

def read_lines(infile):
    with open(infile, 'r', encoding='utf8') as f:
        return f.read().splitlines()
            
def collect_input_meta(reference_files):
    meta = {}
    for ref_file in reference_files:
        file_type = ref_file.split('.')[-1]
        meta[file_type] = read_lines(ref_file)
    return meta

if __name__ == "__main__":
    args = set_args()

    data = {}
    
    # if args.reference_inputs:
    #     eval_set_meta = collect_input_meta(args.reference_inputs)
    # else:
    #     eval_set_meta = None

    data['src_texts'] = read_lines(args.source_texts)
    data['ref_texts'] = read_lines(args.reference_texts)
    # breakpoint()
    
    for i, hyp_texts in enumerate(args.hypothesis_texts):
        data[f'{hyp_texts}'] = read_lines(hyp_texts)
    
    df = pd.DataFrame.from_dict(data)

    # subset dataframe according to randomly drawn indices
    # random_selection = random.sample(range(len(df)), args.n)
    # subset dataframe according to randomly drawn indices
    # df.iloc[random_selection, :]
    df = df.sample(n=args.n, random_state=SEED)
    df['test_set_line_id'] = df.index

    # write to simple jsonl format
    write_to_jsonl_format(df, args.outfile)
            
    
    # with open(args.outfile, 'w', encoding = 'utf8') as outf:

    #     for item in selection:
    #         batch = []
    #         idx = item[0]
    #         id = int(item[1])
        
    #         src_text = srcs['0'][idx]
    #         tgt_text = refs['0'][idx]

    #         # select meta data for patricular item
    #         item_meta = {}
    #         item_meta['eval_id'] = idx
    #         if eval_set_meta:
    #             for k in eval_set_meta:
    #                 item_meta[k] = eval_set_meta[k][idx]

    #         anno_fields = {"fluency": None, "repetition": None, "specif": None, "approp": None, "sent_acc": None, "dom_acc": None}

    #         if args.truecase:
    #             src_text = add_line_breaks(' '.join(caser.get_true_case_from_tokens(src_text.split(), out_of_vocabulary_token_option="as-is")))
    #             tgt_text = add_line_breaks(' '.join(caser.get_true_case_from_tokens(tgt_text.split(), out_of_vocabulary_token_option="as-is")))
    #         else:
    #             src_text = add_line_breaks(src_text)
    #             tgt_text = add_line_breaks(tgt_text)

    #         tgt_item = {
    #                     "text": tgt_text,
    #                     "src": src_text,
    #                     "model_name": "tgt",
    #                     "meta": item_meta,
    #                     "anno": anno_fields
    #                 }

    #         if args.include_tgt:
    #             batch.append(tgt_item)

    #         for model_name in hyps.keys():
                
    #             hyp_text = hyps[model_name][idx]

    #             if args.truecase:
    #                 hyp_text = add_line_breaks(' '.join(caser.get_true_case_from_tokens(hyp_text.split(), out_of_vocabulary_token_option="as-is")))
    #             else:
    #                 hyp_text = add_line_breaks(hyp_text)

    #             # hashed_pair = hash(src_text + ' ' + hyp_text)

    #             # if hashed_pair not in seen_pairs:
                    
    #             #     seen = False
    #             #     seen_pairs.add(hashed_pair)
    #             # else:
    #             #     seen = True

    #             model_item = {
    #                     "text": hyp_text,
    #                     "src": src_text,
    #                     "model_name": model_name,
    #                     "meta": item_meta,
    #                     "anno": anno_fields
    #                 }
    #             batch.append(model_item)

    #         random.shuffle(batch)

    #         for entry in batch:
    #             json_line = json.dumps(entry, ensure_ascii=False)
    #             outf.write(json_line + '\n')


    # # print(f'unique pair count: {len(seen_pairs)}')
    # print(f'Output JSONL file written to {args.outfile}')
