from dask import delayed
from dask.distributed import Client, LocalCluster
import dask, time, statistics
import numpy as np
from mandelbrot_5 import mandelbrot_chunk

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

if __name__ == '__main__':  
    N = 1024
    res = (N, N)
    max_iter = 100
    n_runs = 3
    
    x_dim = (-2.5, 1.0)
    y_dim = (-1.25, 1.25)
    client = Client("tcp://10.92.1.30:8786")
    client.run(lambda: mandelbrot_chunk(row_start=0, row_end=8, N=8, x_dim=x_dim, y_dim=y_dim, max_iter=max_iter))

    times = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        mandelbrot_dask(N=N, x_dim=x_dim, y_dim=y_dim, max_iter=max_iter)
        times.append(time.perf_counter() - t0)
        time_dask = statistics.median(times)
    print(f"Dask local (n_chunks=32): {time_dask:.3f} s")
    client.close()
    #cluster.close()
    
    # times = []
    # for _ in range(n_runs):
    #     t0 = time.perf_counter()
    #     compute_mandelbrot_naive(x_dim=x_dim, y_dim=y_dim, res=res)
    #     times.append(time.perf_counter() - t0)
    #     time_naive = statistics.median(times)
    # print(f'Naive: {time_naive} s')
    # print(f"Speedup: {time_naive/time_dask:.3f} x")

