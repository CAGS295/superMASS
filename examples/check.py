#!/bin/env/python

import numpy as np
import pandas as pd
from superMASS import batched

import tarfile

def load_file(archive, fp):
    """
    Utility function that reads a tar file directly into a numpy array.

    Parameters
    ----------
    archive : str
        The archive file to read.
    fp : str
        The file path of the file to read relative to the archive.


    Returns
    -------
    None if data reading failed or the numpy array of values.
    """
    data = None
    with tarfile.open(archive) as a:
        f = a.extractfile(dict(zip(a.getnames(), a.getmembers()))[fp])
        data = pd.read_csv(f, header=None, names=['reading', ])[
            'reading'].values

    return data


# total number of matches we want returned
top_matches = 5
# length of the subsequence to search in batch processing.
batch_size = 10000

ecg = load_file('data/ecg.tar.gz', 'ecg.txt')
ecg_query = load_file('data/ecg_query.tar.gz', 'ecg_query.txt')

best_indices, best_dists = batched(
    ecg, ecg_query,  top_matches, batch_size)

print(best_indices, best_dists)
