import numpy as np
from numba import njit
from multiprocessing import Pool
import time, os, statistics, matplotlib.pyplot as plt
from pathlib import Path
from mandelbrot_3 import benchmark, compute_mandelbrot_full

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

if __name__ == "__main__":
    # Parameters
    iterations = 100
    N_warmup = 64
    N = 1024
    x_dim = (-2, 1)
    y_dim = (-1.5, 1.5)
    resolution_1 = (64, 64)
    resolution_2 = (1024, 1024)

    # Warm up: First computation doesnt count
    _ = compute_mandelbrot_full(x_dim=x_dim, y_dim=y_dim, res=resolution_1)
    _ = mandelbrot_serial(N=N_warmup, x_dim=x_dim, y_dim=y_dim, max_iter=iterations)

    # Benchmark and plots of numba approach
    t_full, _ = benchmark(compute_mandelbrot_full, x_dim, y_dim, resolution_2, n_runs=iterations)
    t_serial, _ = benchmark(mandelbrot_serial, N, x_dim, y_dim, iterations)
    