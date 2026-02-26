"""
Mandelbrot Set Generator
Author : [ Mathias Jørgensen ]
Course : Numerical Scientific Computing 2026
"""
import numpy as np, time
import matplotlib.pyplot as plt
import time, statistics
import cProfile, pstats
from line_profiler import profile
from numba import jit, njit, int32, complex128
from mandelbrot_1_2 import compute_mandelbrot_naive, compute_mandelbrot_numpy

def benchmark(func, *args, n_runs=3, **kwargs):
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
def compute_mandelbrot_profile(x_dim: tuple[float, float],
                             y_dim: tuple[float, float],
                             res_x: int,
                             res_y: int,
                             max_iter: int = 100):
    
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

# Lecture 3 - Naive implementation with jit & njit
@njit
def mandelbrot_point_numba(c: np.complex128,
                           max_iter: np.int32 = 100) -> np.int32:
    z = 0j
    for n in range(max_iter):
        if z.real*z.real + z.imag*z.imag > 4.0:
            return n
        z = z*z + c
    return max_iter

# Lecture 3 - Naive implementation using numba and mandelbrot_point_numba
def compute_mandelbrot_hybrid(x_dim: tuple[float, float],
                             y_dim: tuple[float, float],
                             res: tuple[int, int]):
    
    # Pulling out variables from tuples
    x_min, x_max = x_dim
    y_min, y_max = y_dim
    res_x, res_y = res

    # Create 1D arrays
    x = np.linspace(x_min, x_max, res_x)
    y = np.linspace(y_min, y_max, res_y)

    #create array for n
    all_n = np.zeros((res_x, res_y), dtype=int)  

    for i in range(res_x):
        for j in range(res_y):
            c = x[i] + 1j * y[j]
            all_n[i, j] = mandelbrot_point_numba(c)
    return all_n

# Lecture 3 - Combination of mandelbrot_point_numba and compute_mandelbrot_numba using njit
@njit
def compute_mandelbrot_full(x_dim: tuple[float, float],
                                   y_dim: tuple[float, float],
                                   res: tuple[int, int],
                                   max_iter: np.int32 = 100):
    
    # Pulling out variables from tuples
    x_min, x_max = x_dim
    y_min, y_max = y_dim
    res_x, res_y = res

    # Create 1D arrays
    x = np.linspace(x_min, x_max, res_x)
    y = np.linspace(y_min, y_max, res_y)

    #create array for n
    result = np.zeros((res_x, res_y), dtype=np.int32)

    for i in range(res_y):
        for j in range(res_x):
            c = x[i] + 1j * y[j]
            z = 0j
            n = 0
            while n < max_iter and z.real*z.real + z.imag*z.imag <= 4.0:
                z = z*z + c
                n += 1
            result[i, j] = n
    return result

@njit
def mandelbrot_numba_typed(x_dim: tuple[float, float],
                           y_dim: tuple[float, float],
                           res: tuple[float, float],
                           max_iter=100, 
                           dtype=np.float64):
    
    # Pulling out variables from tuples
    x_min, x_max = x_dim
    y_min, y_max = y_dim
    res_x, res_y = res

    # Create 1D arrays
    x = np.linspace(x_min, x_max, res_x).astype(dtype)
    y = np.linspace(y_min, y_max, res_y).astype(dtype)
    result = np.zeros((res_y, res_x), dtype=np.int32)

    for i in range(res_y):
        for j in range(res_x):
            c = x[i] + 1j * y[j]
            result[i, j] = mandelbrot_point_numba(c, max_iter)
    return result


if __name__ == "__main__":

    # Parameters
    iterations = 3
    x_dim = (-2, 1)
    y_dim = (-1.5, 1.5)
    resolution_1 = (64, 64)
    resolution_2 = (1024, 1024)

    for dtype in [np.float32, np.float64]:
        t0 = time.perf_counter()
        mandelbrot_numba_typed(x_dim, y_dim, resolution_2, dtype=dtype)
        print(f'{dtype.__name__}: {time.perf_counter()-t0:.3f}s')

    r32 = mandelbrot_numba_typed(x_dim, y_dim, resolution_2, dtype=np.float32)
    r64 = mandelbrot_numba_typed(x_dim, y_dim, resolution_2, dtype=np.float64)

    fig, axes = plt.subplots(1, 2, figsize=(12,4))
    for ax, result, title in zip(axes, [r32, r64], ['float32', 'float64 (ref)']):
        ax.imshow(result, cmap='hot')
        ax.set_title(title)
        ax.axis('off')

    plt.savefig('precision_comparison.png', dpi=150)

    print(f'Max diff float32 vs float64: {np.abs(r32-r64).max()}')
    
    """
    Results:
    float32: 1.477s
    float64: 0.317s
    Max diff float32 vs float64: 33
    """

    





    
    
