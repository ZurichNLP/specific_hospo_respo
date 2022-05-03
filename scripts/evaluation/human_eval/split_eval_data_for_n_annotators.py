#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import random
import json
from pathlib import Path

SEED=42

def set_args():
    ap = argparse.ArgumentParser()
    ap.add_argument('--infile', type=str, required=True)
    ap.add_argument('--outdir', type=str, required=True)
    ap.add_argument('-n', type=int, required=False, default=5, help='number of annotators involved')
    ap.add_argument('--total', type=int, required=False, default=100, help='number of items to be annotated by each annotator')
    ap.add_argument('--shared_items', type=int, required=False, default=25, help='number of items that all annotators see')
    return ap.parse_args()

def iter_lines(infile):
    with open(infile, 'r', encoding='utf-8') as f:
        for line in f:
            line = json.loads(line.strip())
            yield line
    
if __name__ == '__main__':

    args = set_args()
    # collect all data
    data = list(iter_lines(args.infile))
    shared_partition = data[:args.shared_items]
    data = data[args.shared_items:]
    # breakpoint()
    # chunks = [data[x:x+100] for x in range(0, len(data), args.total-args.shared_items)]

    for i in range(args.n):
        annotator_outfile = Path(args.outdir) / f'annotator_{i}.jsonl'
        annotator_chunk = shared_partition + data[:args.total-args.shared_items]
        data = data[args.total-args.shared_items:]
        # random.seed(SEED)
        # random.shuffle(annotator_chunk)
        with open(annotator_outfile, 'w', encoding='utf8') as outf:
            for item in annotator_chunk:
                outf.write(f'{json.dumps(item)}\n')
