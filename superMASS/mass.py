from .mass_rs import batched as __batched, mass as __mass

import numpy as np


def batched(series: np.array, query: np.array, top_matches: int, batch_size: int) -> (np.array, np.array):
    """
    MASS3 batch is a batch version of MASS3 that reduces overall memory usage,
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

    Returns:
        (np.array,np.array): 
    """
    return __batched(series, query, batch_size, top_matches)


def mass3(series: np.array, query: np.array) -> np.array:
    """MASS3 chunks computation into slices power of two to speed up the fft internally used for quick multiplication.

    See refs:
        https://www.cs.unm.edu/~mueen/FastestSimilaritySearch.html
        https://www.cs.unm.edu/~mueen/MASS_V3.m

    Args:
        series (np.array): A one dimensional array where patterns are sought.
        query (np.array): A one dimensional array. This is the pattern strided along [series].

    Returns:
        np.array: A one dimensional array with the distances aligned to the beggining ofthe query,
        i.e. distance[i] belongs to the subsequence starting at series[i].

    Example:
    '''
    import numpy as np
    from superMASS import mass3

    a = np.array([0.0, 1.0, 2., 3., 5., 6.])
    b = np.array([2.0, 3.0])
    c = mass3(a, b)
    '''
    """

    return __mass(series, query)
