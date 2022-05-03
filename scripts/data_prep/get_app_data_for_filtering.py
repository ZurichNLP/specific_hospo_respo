#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pathlib import Path
from tqdm import tqdm

data_dir = Path('../data/apps/')
output_dir = data_dir / 'src-tgt'

output_dir.mkdir(parents=True, exist_ok=True)

# read in train/test/valid files and write src-tgt outputs for BART finetuning
for split in ['train', 'test', 'valid']:
    csv_file = Path(data_dir) / f'rrgen_{split}_data.txt'
    src_file = output_dir / f'{split}.src'
    tgt_file = output_dir / f'{split}.tgt'
    
    with open(
        csv_file, 'r', encoding='utf8') as inf, open(
            src_file, 'w', encoding='utf8') as srcf, open(
                tgt_file, 'w', encoding='utf8') as tgtf:

            for i, line in tqdm(enumerate(inf, 1)):
        
                line_list = line.split('***')

                if len(line_list) < 8:    # check term length
                    print(f'line {i} missing fields: {line}')

                review = line_list[4]
                response = line_list[5]
                if review != '' and response != '':
                    srcf.write(f'{review}\n')
                    tgtf.write(f'{response}\n')


# prepare train/test/valid data from validation split for lm finetuning
valid_tgt = output_dir / 'valid.tgt'

output_dir = data_dir / 'tgt'
output_dir.mkdir(parents=True, exist_ok=True)
with open(valid_tgt, 'r', encoding='utf8') as inf:
    with open(output_dir / 'vtrain.tgt', 'w', encoding='utf8') as train_f:
        # with open(output_dir / 'vtest.tgt', 'w', encoding='utf8') as test_f:
        with open(output_dir / 'vval.tgt', 'w', encoding='utf8') as val_f:
            for i, line in tqdm(enumerate(inf)):
                line = '<|text|>' + line.strip() + '<|endoftext|>\n'
                if i % 10 == 0:
                    val_f.write(line)
                else:
                    train_f.write(line)

print('done!')