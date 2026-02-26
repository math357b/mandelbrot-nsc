"""
Mandelbrot Set Generator
Author : [ Mathias Jørgensen ]
Course : Numerical Scientific Computing 2026
"""
import numpy as np
import matplotlib.pyplot as plt
import time, statistics
import cProfile, pstats
from line_profiler import profile

def benchmark(func, *args, n_runs=3):
    """Time func, return median of n_runs"""
    times = []
    
    for _ in range(n_runs):
        t0 = time.perf_counter()
        result = func(*args)
        times.append(time.perf_counter() - t0)
    median_t = statistics.median(times)
    print(f'Median: {median_t:.4f}s'
          f'(min={min(times):.4f}, max={max(times):.4f})')
    return median_t, result

# Lecture 3 - Naive Implementation with profile
@profile     
def compute_mandelbrot_naive(x_dim = tuple[float, float],
                             y_dim = tuple[float, float],
                             res_x = int,
                             res_y = int,
                             max_iter=100):
    
    # Pulling out variables from tuples
    x_min, x_max = x_dim
    y_min, y_max = y_dim

    # Create 1D arrays
    x = np.linspace(x_min, x_max, res_x)
    y = np.linspace(y_min, y_max, res_y)
    result = np.zeros((res_y, res_x), dtype=int)

    for i in range(res_y):
        for j in range(res_x):
            c = x[j] + 1j * y[i]
            z = 0
            for n in range(max_iter):
                if abs(z) > 2:
                    result[i, j] = n
                    break
                z = z*z + c
            else:
                result[i, j] = max_iter
    return result

if __name__ == "__main__":

    # Parameters
    iterations = 3
    x_dim = (-2, 1)
    y_dim = (-1.5, 1.5)
    resolution = (512, 512)


    compute_mandelbrot_naive(x_dim=x_dim,
                             y_dim=y_dim,
                             res_x=resolution[0],
                             res_y=resolution[1])