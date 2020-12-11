from .mass_rs import batched as __batched
import numpy as np


def batched(series: np.array, query: np.array, top_matches: int, batch_size: int, n_jobs: int) -> (np.array, np.array):
    """
    MASS2 batch is a batch version of MASS2 that reduces overall memory usage,
    provides parallelization and enables you to find top K number of matches
    within the time series. The goal of using this implementation is for very
    large time series similarity search. The returned results are not sorted
    by distance. So you will need to find the top match with np.argmin() or
    sort them yourself.

    Args:
        series (np.array): Array of floats that represents a time series
        query (np.array): a query for similarity search strided by one along [series]
        batch_size (int): split the search in this many slices; useful for parallelization
        top_matches (int): top nth matches to return.
        n_jobs (int): parallelize search in this many jobs

    Returns:
        (np.array,np.array): 
    """
    return __batched(series, query, batch_size, top_matches, n_jobs)
