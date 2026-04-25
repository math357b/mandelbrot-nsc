import numpy as np
from numba import njit
from multiprocessing import Pool
import time
import os
import statistics
import matplotlib.pyplot as plt

@njit(cache=True)
def mandelbrot_pixel(c_real: float,
                     c_imag: float,
                     max_iter: int):
    
    '''Compute the Mandelbrot escape iteration count for a single pixel.

    This function evaluates the iteration z_{n+1} = z_n^2 + c for a given
    complex point c = c_real + i*c_imag, starting from z = 0, and returns
    the iteration at which the magnitude of z exceeds 2. If the point does
    not escape within 'max_iter' iterations, 'max_iter' is returned.

    Parameters
    ----------
    c_real : float
        Real part of the complex parameter c.
    c_imag : float
        Imaginary part of the complex parameter c.
    max_iter : int
        Maximum number of iterations to test for divergence.

    Returns
    -------
    i : int
        Number of iterations before divergence (|z| > 2).
        Returns 'max_iter' if the point does not diverge.
    '''
        
    z_real = z_imag = 0.0

    for i in range(max_iter):
        zr2 = z_real*z_real
        zi2 = z_imag*z_imag
        
        if zr2 + zi2 > 4:
            return i
        
        z_imag = 2.0*z_real*z_imag + c_imag
        z_real = zr2 - zi2 + c_real
    return max_iter

@njit(cache=True)
def mandelbrot_chunk(row_start: int,
                     row_end: int,
                     N: int,
                     x_dim: tuple[float, float],
                     y_dim: tuple[float, float],
                     max_iter: int):
    
    '''Compute a rectangular chunk of the Mandelbrot set using Numba.

    This function evaluates a subset of rows in a Mandelbrot grid, allowing
    for manual parallelization by splitting the computation into chunks.

    Parameters
    ----------
    row_start : int
        Starting row index (inclusive of the chunk).
    row_end : int
        Ending row index (exclusive of the chunk).
    N : int
        Number of columns (resolution in x-direction).
    x_dim : tuple of float
        (x_min, x_max) range of the real axis.
    y_dim : tuple of float
        (y_min, y_max) range of the imaginary axis.
    max_iter : int
        Maximum number of iterations for divergence testing.

    Returns
    -------
    out : ndarray of shape (row_end - row_start, N)
        2D array containing escape iteration counts for the specified chunk
        of the Mandelbrot set.
    '''
    
    x_min, x_max = x_dim
    y_min, y_max = y_dim

    out = np.empty((row_end - row_start, N), dtype=np.int32)
    dx = (x_max - x_min) / N
    dy = (y_max - y_min) / N
    for r in range(row_end - row_start):
        c_imag = y_min + (r + row_start) * dy
        for col in range(N):
            out[r, col] = mandelbrot_pixel(c_real=x_min+col*dx, c_imag=c_imag, max_iter=max_iter)
    return out

def mandelbrot_serial(N: int,
                      x_dim: tuple[float, float],
                      y_dim: tuple[float, float],
                      max_iter=100):
    
    '''Compute the full Mandelbrot set using a serial chunk-based implementation.

    This function is a wrapper around 'mandelbrot_chunk' that
    computes the entire grid in a single call (no parallelization), treating
    the full image as one chunk.

    Parameters
    ----------
    N : int
        Resolution of the grid (assumes N x N square output).
    x_dim : tuple of float
        (x_min, x_max) range of the real axis.
    y_dim : tuple of float
        (y_min, y_max) range of the imaginary axis.
    max_iter : int, optional
        Maximum number of iterations for divergence testing (default is 100).

    Returns
    -------
    out : ndarray of shape (N, N)
        2D array containing escape iteration counts for the full Mandelbrot
        set over the specified domain.
    '''

    return mandelbrot_chunk(row_start=0, row_end=N, N=N, x_dim=x_dim, y_dim=y_dim, max_iter=max_iter)

def _worker(args):
    '''Worker function for multiprocessing Mandelbrot computation.

    This function unpacks arguments and delegates computation to
    'mandelbrot_chunk', allowing each process to compute a portion
    of the Mandelbrot set independently.

    Parameters
    ----------
    args : tuple
        Arguments for 'mandelbrot_chunk', expected as:
        (row_start, row_end, N, x_dim, y_dim, max_iter)

    Returns
    -------
    ndarray
        A 2D array containing escape iteration counts for the assigned chunk.
    '''
    return mandelbrot_chunk(*args)

def mandelbrot_parallel(N: int,
                        x_dim: tuple[float, float],
                        y_dim: tuple[float, float],
                        max_iter: int = 100,
                        num_workers: int = 4,
                        n_chunks=None,
                        pool=None):
    
    '''
    Compute the Mandelbrot set in parallel using multiprocessing,
    with optional external pool reuse and warm-up initialization.

    The image is divided into row-wise chunks and distributed across
    multiple worker processes. Each worker computes a portion of the
    Mandelbrot set using 'mandelbrot_chunk', and results are combined
    into a full grid.

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
    num_workers : int, optional
        Number of worker processes to use if a pool is not provided (default is 4).
    n_chunks : int or None, optional
        Number of chunks to split the work into. If None, defaults to
        'num_workers'.
    pool : multiprocessing.Pool or None, optional
        Existing multiprocessing pool. If provided, it is reused and no new
        pool is created. If None, a temporary pool is created.

    Returns
    -------
    result : ndarray of shape (N, N)
        2D array containing escape iteration counts for the full Mandelbrot
        set over the specified domain.
    '''

    if n_chunks is None:
        n_chunks = num_workers
    chunk_size = max(1, N // n_chunks)
    chunks = []
    row = 0
    while row < N:
        row_end = min(row + chunk_size, N) # determine end of the row
        chunks.append((row, row_end, N, x_dim, y_dim, max_iter))
        row = row_end

    if pool is not None:        # caller manages Pool; skip startup + warm-up
        return np.vstack(pool.map(_worker, chunks))
    tiny = [(0, 8, 8, x_dim, y_dim, max_iter)]            
    with Pool(processes=num_workers) as p:
        p.map(_worker, tiny)
        parts = p.map(_worker, chunks)
                                # warm-up: load JIT cache in workers
    return np.vstack(parts)

if __name__ == "__main__":
    
    # Parameters
    N = 1024
    x_dim = (-2.5, 1.0)
    y_dim = (-1.25, 1.25)
    max_iter = 100
    num_workers = 4
    num_runs = 3

    workers_list = []
    speedups = []

    # Serial baseline (Numba already warm after M1 warm-up)
    times = []
    for _ in range(num_runs):
        start_time = time.perf_counter()
        mandelbrot_serial(N=N, x_dim=x_dim, y_dim=y_dim, max_iter=max_iter)
        times.append(time.perf_counter() - start_time)
    t_serial = statistics.median(times)

    for num_workers in range(1, os.cpu_count() + 1):
        chunk_size = max(1, N // num_workers)
        chunks = []
        row = 0
        while row < N:
            row_end = min(row + chunk_size, N) # determine end of the row
            chunks.append((row, row_end, N, x_dim, y_dim, max_iter))
            row = row_end

        with Pool(processes=num_workers) as pool:
            pool.map(_worker, chunks)
            times = []
            for _ in range(num_runs):
                t0 = time.perf_counter()
                np.vstack(pool.map(_worker, chunks))
                times.append(time.perf_counter() - t0)
        t_parallel = statistics.median(times)
        speedup = t_serial / t_parallel

        # Append workers and speedups for plotting
        workers_list.append(num_workers)
        speedups.append(speedup)

        # print table of workers, speedup and efficiency
        print(f'{num_workers:2d} workers | '
              f'parallel time: {t_parallel:.4f}s | '
              f'speedup: {speedup:.2f}x | '
              f'efficiency: {speedup/num_workers:.2f}')
    
    # Plot worker vs. speedup
    #plot_worker_speedup(title="Mandelbrot", workers=workers_list, speedup=speedups)
    #plt.savefig("figures/Mandelbrot_worker_speedup.png")

    # Milestone 1 testing:
    result_serial = mandelbrot_serial(N=N, x_dim=x_dim, y_dim=y_dim)
    result_parallel = mandelbrot_parallel(N=N, x_dim=x_dim, y_dim=y_dim, num_workers=num_workers)

    fig, axes = plt.subplots(1, 2, figsize=(16,6))
    axes[0].imshow(result_serial, extent=[x_dim[0], x_dim[1], y_dim[0], y_dim[1]], cmap='inferno', origin='lower', aspect='equal')
    axes[0].set_title('Serial Mandelbrot')
    axes[0].set_xlabel('Re(c)')
    axes[0].set_ylabel('Im(c)')

    axes[1].imshow(result_parallel, extent=[x_dim[0], x_dim[1], y_dim[0], y_dim[1]], cmap='inferno', origin='lower', aspect='equal')
    axes[1].set_title('Parallel Mandelbrot')
    axes[1].set_xlabel('Re(c)')
    axes[1].set_ylabel('Im(c)')
    plt.show()

    # Milestone 2
    
    
        
        

    