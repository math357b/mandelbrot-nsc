from dask import delayed
from dask.distributed import Client, LocalCluster
import dask, time, statistics
import numpy as np
from lecture_py_files.mandelbrot_5 import mandelbrot_chunk

def mandelbrot_dask(N: int,
                    x_dim: tuple[float, float],
                    y_dim: tuple[float, float],
                    max_iter: int = 100,
                    n_chunks: int = 32):
    
    chunk_size = max(1, N // n_chunks)
    tasks = []
    row = 0

    while row < N:
        row_end = min(row + chunk_size, N)
        tasks.append(delayed(mandelbrot_chunk)(row, row_end, N, x_dim, y_dim, max_iter))
        row = row_end
    
    parts = dask.compute(*tasks)

    return np.vstack(parts)
