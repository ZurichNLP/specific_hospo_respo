#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import sys
from pathlib import Path


in_file = sys.argv[1] # test set
out_file = sys.argv[2]
n_items = 0 #int(sys.argv[3])

# Path(outdir).mkdir(parents=True, exist_ok=False)

with open(in_file, 'r', encoding='utf8') as inf:
    with open(out_file, 'w', encoding='utf8') as outf:
        for line in inf:
            for n in range(6):
                outf.write(f'ppl_{n}: {line}')
            
            n_items += 1

print(f'{n_items} written to {out_file} ...')
