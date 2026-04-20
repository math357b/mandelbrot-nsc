import numpy as np
from numba import njit
from multiprocessing import Pool
import time, os, statistics, matplotlib.pyplot as plt
from pathlib import Path
from lecture_py_files.mandelbrot_3 import benchmark, compute_mandelbrot_full
from monte_carlo_example import plot_worker_speedup

# MP2-M1: Creates a serial implementation, equal to the one in L3  
@njit
def mandelbrot_pixel(c_real,
                     c_imag,
                     max_iter):
    z_real = z_imag = 0.0
    for i in range(max_iter):
        zr2 = z_real*z_real
        zi2 = z_imag*z_imag
        if zr2 + zi2 > 4: return i
        z_imag = 2.0*z_real*z_imag + c_imag
        z_real = zr2 - zi2 + c_real
    return max_iter

@njit
def mandelbrot_chunk(row_start: int,
                     row_end: int,
                     N: int,
                     x_dim: tuple[float, float],
                     y_dim: tuple[float, float],
                     max_iter: int):
    
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

    return mandelbrot_chunk(row_start=0, row_end=N, N=N, x_dim=x_dim, y_dim=y_dim, max_iter=max_iter)

def _worker(args):
    return mandelbrot_chunk(*args)

def mandelbrot_parallel(N: int,
                        x_dim: tuple[float, float],
                        y_dim: tuple[float, float],
                        max_iter: int = 100,
                        num_workers: int = 4):

    chunk_size = max(1, N // num_workers) # Divide work into chunks
    chunks = []
    row = 0
    while row < N:
        row_end = min(row + chunk_size, N) # determine end of the row
        chunks.append((row, row_end, N, x_dim, y_dim, max_iter))
        row = row_end

    with Pool(processes=num_workers) as pool:
        pool.map(_worker, chunks)
        parts = pool.map(_worker, chunks)

    return np.vstack(parts)

if __name__ == "__main__":
    
    # Parameters
    N = 1024
    x_dim = (-2.5, 1.0)
    y_dim = (-1.25, 1.25)
    num_workers = 4

    # Milestone 2 testing:
    result = mandelbrot_parallel(N=N, x_dim=x_dim, y_dim=y_dim, num_workers=num_workers)

    fig, ax = plt.subplots(figsize=(8,6))
    ax.imshow(result, extent=[x_dim[0], x_dim[1], y_dim[0], y_dim[1]], cmap='inferno', origin='lower', aspect='equal')
    ax.set_xlabel('Re(c)')
    ax.set_ylabel('Im(c)')
    out = Path(__file__).parent / 'figures/mandelbrot_parallel.png'
    fig.savefig(out, dpi=150)
    print(f'Saved: {out}')
    

    """
    # Parameters
    N = 1024
    x_dim = (-2.5, 1.0)
    y_dim = (-1.25, 1.25)
    max_iter = 100
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
    plot_worker_speedup(title="Mandelbrot", workers=workers_list, speedup=speedups)
    plt.savefig("figures/Mandelbrot_worker_speedup.png")
    """
        
        
