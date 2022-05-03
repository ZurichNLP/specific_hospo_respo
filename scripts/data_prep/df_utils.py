#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pathlib import Path
from typing import Dict, List

SEED = 42

col_name_outfile_mapping = {
    'rrgen_id': 'rrgen_id', 
    'review_clean': 'review', # normal review 
    'response_clean': 'response',  # normal response
    'rating': 'rating', # normal review rating
    'establishment': 'establishment',
}

def write_file(series, outfile):
    with open(outfile, 'w', encoding='utf8') as f:
        for line in series.to_list():
            f.write(f'{line}\n')
    return

def generate_fairseq_input_files(df,
                                 outdir: str,
                                 col_name_outfile_mapping: Dict = col_name_outfile_mapping,
                                 split_col: str = 'split',
                                 n: int = 0):
    """
    Generates multiple individual files (one per column).
    For each split (train/test/valid) lines in each output file must correspond with each other!
    """
    for split in df[split_col].unique():  
        
        split_df = df[df[split_col] == split]
        
        # shuffle train set - mainly required after upsampling!
        if split == 'train':
            split_df = split_df.sample(frac=1, random_state=SEED)
        
        if n: # just take a head of dataframe
            if split == 'train':
                split_df = split_df.head(n)
            else:
                split_df = split_df.head(int(n*0.1))

        print(f'{split} split has length: {len(split_df)}')

        for k, v in col_name_outfile_mapping.items():
            write_file(split_df[k], outdir / f'{split}.{v}')
        
    print('Done!')
    return


def inspect_scores(df):
    print(len(df))
    print()
    print(df['split'].value_counts())
    for col_name in df.columns:
        if col_name.startswith('score:'):
            print(df[col_name].describe())
            print()
    return

if __name__ == "__main__":
    pass