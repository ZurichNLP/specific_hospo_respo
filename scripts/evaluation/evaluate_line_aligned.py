#!/usr/bin/env python
# coding: utf-8

import argparse
from typing import List, Dict, Optional, Tuple
import pandas as pd
from pathlib import Path

import sacrebleu
from rrgen_evaluations import *

tokenize = sacrebleu.tokenizers.tokenizer_13a.Tokenizer13a()

def set_args():
    ap = argparse.ArgumentParser()
    ap.add_argument('hyp_files', type=str, help='path to longmbart training dir (containing files _val_out_checkpoint_*')
    # ap.add_argument('--ref_dir', type=str, required=False, help='path to original input source file e.g. re_test.review(.sp)')
    ap.add_argument('--src_file', type=str, required=True, help='path to original input source file e.g. re_test.review(.sp)')
    ap.add_argument('--ref_file', type=str, required=True, help='path to original target file e.g. re_test.response(.sp)')
    ap.add_argument('--sp_model', type=str, required=False, help='path to spm model to use for decoding')
    ap.add_argument('--domain_ref', type=str, required=False, default=None, help='path to domain ground truth labels, e.g. re_test.domain')
    ap.add_argument('--rating_ref', type=str, required=False, default=None, help='path to rating ground review rating labels, e.g. re_test.rating')
    ap.add_argument('--source_ref', type=str, required=False, default=None, help='path to source ground review rating labels, e.g. re_test.source')
    ap.add_argument('--compute_sts', action='store_true', required=False, help='use if need to compute repetition metric with sbert')
    ap.add_argument('--read_in_tokenized', type=bool, required=False, default=True, help='for scoring raw inputs, lines are tokenized as they are read in using sacrebleu tokenizer13a.')
    return ap.parse_args()

def read_lines(infile: str, read_in_tokenized: bool = True) -> List[str]:
    """expects OSPL format file without sentencepiece encoding"""
    lines = []
    with open(infile, 'r', encoding='utf8') as f:
        for line in f:
            if read_in_tokenized:
                lines.append(tokenize(line.strip()))
            else:
                lines.append(line.strip())
        return lines
    
if __name__ == '__main__':
    args = set_args()

    # if args.src_file:
    srcs = read_lines(args.src_file, read_in_tokenized=args.read_in_tokenized)
    refs = read_lines(args.ref_file, read_in_tokenized=args.read_in_tokenized)

    # print(len(srcs))

    domain_refs, rating_refs, source_refs = None, None, None

    if args.domain_ref:
        domain_refs = read_lines(args.domain_ref)
        assert len(domain_refs) == len(srcs)
    if args.rating_ref:
        rating_refs = read_lines(args.rating_ref)
        assert len(rating_refs) == len(srcs)
    if args.source_ref:
        source_refs = read_lines(args.source_ref)
        assert len(source_refs) == len(srcs)

    if Path(args.hyp_files).is_dir():
        # if directory path is given, assumed to evaluate
        # validation outputs.
        all_scores = []
        for i in range(20):
            hyp_file = Path(args.hyp_files) / str('_val_out_checkpoint_'+str(i))
            hyps = read_lines(hyp_file, read_in_tokenized=args.read_in_tokenized)   
            scores = run_eval(srcs, refs, hyps, hyp_file, domain_refs, rating_refs, source_refs, compute_sts_metrics=args.compute_sts, verbose=False)
            all_scores.append(scores)
        df = pd.concat(all_scores)
        print(df.to_csv())
            
    else:
        hyp_file = args.hyp_files
        hyps = read_lines(hyp_file, read_in_tokenized=args.read_in_tokenized)   
        scores = run_eval(srcs, refs, hyps, hyp_file, domain_refs, rating_refs, source_refs, compute_sts_metrics=args.compute_sts, verbose=True)


