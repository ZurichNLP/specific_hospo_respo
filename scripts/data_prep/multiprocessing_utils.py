#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import multiprocessing
import numpy as np
import pandas as pd
from tqdm import tqdm
import time


def parallelise(func, iterable, n_cores):
    """
    Simple parallelisation function.

    func: function to be applied
    iterable: list-type object for processing
    """
    
    print('Running jobs on {} CPU(s)'.format(n_cores))
    
    START_TIME = time.time()
    
    with multiprocessing.Pool(n_cores) as p:
#         result = p.map(func, iterable)
        result = list(tqdm(p.imap(func, iterable), total=len(iterable)))
        p.close()
        p.join()
        
    END_TIME = time.time()
    t = END_TIME - START_TIME
    print('Time taken: {:.2f} seconds'.format(t))

    return result


def parallelize_dataframe(df, func, n_cores=10):
    """
    Splits DF in chunks for applying function to each chunk in parallel

    For large DFs, limit n_cores to avoid OOM error.

    Source: https://towardsdatascience.com/make-your-own-super-pandas-using-multiproc-1c04f41944a1

    df: Pandas DF for processing
    func: function to be applied
    n_cores: number of jobs to be spawned

    """

    print('Running jobs on {} CPU(s)'.format(n_cores))

    START_TIME = time.time()

    df_splits = np.array_split(df, n_cores)
    with multiprocessing.Pool(n_cores) as p:
        result = list(tqdm(p.imap(func, df_splits), total=len(df_splits)))
        p.close()
        p.join()

    df = pd.concat(result)

    END_TIME = time.time()
    t = END_TIME - START_TIME
    print('Time taken: {:.2f} seconds'.format(t))

    return df

if __name__ == "__main__":
    pass
