from dask import delayed
#from dask.distributed import Client, LocalCluster
import dask
#import time
#import statistics
import numpy as np
from lecture_py_files.mandelbrot_5 import mandelbrot_chunk

def mandelbrot_dask(N: int,
                    x_dim: tuple[float, float],
                    y_dim: tuple[float, float],
                    max_iter: int = 100,
                    n_chunks: int = 32):
    
    '''
    Compute the Mandelbrot set using Dask-based parallel task scheduling.

    The image is divided into row-wise chunks, each of which is executed
    as a delayed Dask task. The tasks are then computed in parallel and
    combined into a full grid.

    Parameters
    ----------
    N : int
        Resolution of the grid (assumes an N x N square output).
    x_dim : tuple of float
        (x_min, x_max) range of the real axis.
    y_dim : tuple of float
        (y_min, y_max) range of the imaginary axis.
    max_iter : int, optional
        Maximum number of iterations for divergence testing (default is 100).
    n_chunks : int, optional
        Number of chunks to divide the computation into (default is 32).

    Returns
    -------
    result : ndarray of shape (N, N)
        2D array containing escape iteration counts for the full Mandelbrot
        set over the specified domain.
    '''
    
    chunk_size = max(1, N // n_chunks)
    tasks = []
    row = 0

    while row < N:
        row_end = min(row + chunk_size, N)
        tasks.append(delayed(mandelbrot_chunk)(row, row_end, N, x_dim, y_dim, max_iter))
        row = row_end
    
    parts = dask.compute(*tasks)

    return np.vstack(parts)
