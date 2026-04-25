"""
Mandelbrot Set Generator
Author : [ Mathias Jørgensen ]
Course : Numerical Scientific Computing 2026
"""
import numpy as np
import matplotlib.pyplot as plt
#import time
#import statistics
#import cProfile
#import pstats
from line_profiler import profile
from numba import njit #, jit, prange
#from lecture_py_files.mandelbrot_1_2 import compute_mandelbrot_naive, compute_mandelbrot_numpy, benchmark

# Lecture 3 - Naive Implementation with profile
@profile     
def compute_mandelbrot_profile(x_dim: tuple[float, float],
                               y_dim: tuple[float, float],
                               res: tuple[int, int],
                               max_iter: int = 100):
    '''Compute a Mandelbrot set grid using a nested-loop implementation
    optimized for line-by-line profiling.

    This function evaluates the escape iteration count for each point in a
    2D grid of the complex plane using explicit Python loops.

    Parameters
    ----------
    x_dim : tuple of float
        (x_min, x_max) range of the real axis.
    y_dim : tuple of float
        (y_min, y_max) range of the imaginary axis.
    res : tuple of int
        (res_x, res_y) number of points on both axis.
    max_iter : int, optional
        Maximum number of iterations for divergence testing (default is 100).

    Returns
    -------
    result : ndarray of shape (res_y, res_x)
        2D array where each element contains the number of iterations
        before divergence for the corresponding point. Points that do not
        diverge within 'max_iter' iterations are assigned 'max_iter'.
    '''
    
    # Pulling out variables from tuples
    x_min, x_max = x_dim
    y_min, y_max = y_dim
    res_x, res_y = res

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
    
    '''Compute the escape iteration count for a point in the Mandelbrot set
    using Numba JIT compilation.

    This function applies the recurrence relation
    z_{n+1} = z_n^2 + c starting from z_0 = 0, and returns the iteration
    at which the magnitude of z exceeds 2. If the point does not escape
    within 'max_iter' iterations, 'max_iter' is returned.

    Parameters
    ----------
    c : np.complex128
        Complex number representing the point in the complex plane.
    max_iter : np.int32, optional
        Maximum number of iterations for divergence testing (default is 100).

    Returns
    -------
    n : np.int32
        Number of iterations before divergence (|z| > 2).
        Returns 'max_iter' if the point does not diverge.
    '''

    z = 0j
    for n in range(max_iter):
        if z.real*z.real + z.imag*z.imag > 4.0:
            return n
        z = z*z + c
    return max_iter

# Lecture 3 - Naive implementation using numba and mandelbrot_point_numba
def compute_mandelbrot_hybrid(x_dim: tuple[float, float],
                              y_dim: tuple[float, float],
                              res: tuple[int, int],
                              max_iter: int = 100):
    
    '''Compute a Mandelbrot set grid using a hybrid approach combining
    Python loops and a Numba-accelerated point function.

    Each point in the complex plane is evaluated using
    'mandelbrot_point_numba', which is JIT-compiled for performance,
    while the outer grid traversal is performed using Python loops.

    Parameters
    ----------
    x_dim : tuple of float
        (x_min, x_max) range of the real axis.
    y_dim : tuple of float
        (y_min, y_max) range of the imaginary axis.
    res : tuple of int
        (res_x, res_y) number of points in the x and y directions.
    max_iter : int, optional
        Maximum number of iterations for divergence testing (default is 100).

    Returns
    -------
    all_n : ndarray of shape (res_x, res_y)
        2D array where each element contains the number of iterations
        before divergence for the corresponding point. Points that do not
        diverge within 'max_iter' iterations are assigned 'max_iter'.
    '''

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
            all_n[i, j] = mandelbrot_point_numba(c=c, max_iter=max_iter)
    return all_n

# Lecture 3 - Combination of mandelbrot_point_numba and compute_mandelbrot_numba using njit
@njit
def compute_mandelbrot_full(x_dim: tuple[float, float],
                                   y_dim: tuple[float, float],
                                   res: tuple[int, int],
                                   max_iter: np.int32 = 100):
    
    '''Fully Numba-accelerated computation of the Mandelbrot set grid.

    This function computes the escape iteration count for each point in a
    2D grid of the complex plane using a fully compiled Numba implementation.
    Both the outer grid traversal and inner iteration loop are optimized.

    Parameters
    ----------
    x_dim : tuple of float
        (x_min, x_max) range of the real axis.
    y_dim : tuple of float
        (y_min, y_max) range of the imaginary axis.
    res : tuple of int
        (res_x, res_y) number of points in the x and y directions.
    max_iter : np.int32, optional
        Maximum number of iterations for divergence testing (default is 100).

    Returns
    -------
    result : ndarray of shape (res_y, res_x)
        2D array where each element contains the number of iterations
        before divergence for the corresponding point. Points that do not
        diverge within 'max_iter' iterations are assigned 'max_iter'.
    '''
    
    # Pulling out variables from tuples
    x_min, x_max = x_dim
    y_min, y_max = y_dim
    res_x, res_y = res

    # Create 1D arrays
    x = np.linspace(x_min, x_max, res_x)
    y = np.linspace(y_min, y_max, res_y)

    #create array for n
    result = np.zeros((res_y, res_x), dtype=np.int32)

    for i in range(res_y):
        for j in range(res_x):
            c = x[j] + 1j * y[i]
            z = 0j
            n = 0
            while n < max_iter and z.real*z.real + z.imag*z.imag <= 4.0:
                z = z*z + c
                n += 1
            result[i, j] = n
    return result

# Lecture 3 - Comparison of different datatypes (float32 vs float64)
@njit
def mandelbrot_numba_typed(x_dim: tuple[float, float],
                           y_dim: tuple[float, float],
                           res: tuple[int, int],
                           max_iter=100, 
                           dtype=np.float64):
    '''Compute a Mandelbrot set grid using a typed NumPy + Numba hybrid approach.

    This function generates a discretized complex plane and computes the
    escape iteration count for each point using a Numba-compiled point function.

    Parameters
    ----------
    x_dim : tuple of float
        (x_min, x_max) range of the real axis.
    y_dim : tuple of float
        (y_min, y_max) range of the imaginary axis.
    res : tuple of int
        (res_x, res_y) number of points along x and y axes.
    max_iter : int, optional
        Maximum number of iterations for divergence testing (default is 100).
    dtype : data-type, optional
        Floating-point precision used for coordinate arrays (default is np.float64).

    Returns
    -------
    result : ndarray of shape (res_y, res_x)
        2D array where each element contains the number of iterations
        before divergence. Points that do not diverge are assigned 'max_iter'.
    '''
    
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
            c = x[j] + 1j * y[i]
            result[i, j] = mandelbrot_point_numba(c, max_iter)
    return result


if __name__ == "__main__":
    # Parameters
    iterations = 100
    N = 1024
    x_dim = (-2, 1)
    y_dim = (-1.5, 1.5)
    resolution_1 = (64, 64)
    resolution_2 = (1024, 1024)

    """
    # Warm up: First computation doesnt count
    _ = compute_mandelbrot_full(x_dim=x_dim, y_dim=y_dim, res=resolution_1)

    # Benchmark and plots of numba approach
    t_full, _ = benchmark(compute_mandelbrot_full, x_dim, y_dim, resolution_2, n_runs=iterations)

    result_numba = compute_mandelbrot_full(x_dim=x_dim, y_dim=y_dim, res=resolution_2, max_iter=iterations)

    fig, ax = plt.subplots(figsize=(8,6))
    ax.imshow(result_numba, extent=[x_dim[0], x_dim[1], y_dim[0], y_dim[1]], cmap='inferno', origin='lower', aspect='equal')
    ax.set_title(' Serial Mandelbrot')
    ax.set_xlabel('Re(c)')
    ax.set_ylabel('Im(c)')
    plt.show()
    """

    result_numba_hybrid = compute_mandelbrot_hybrid(x_dim, y_dim, resolution_2)

    fig, ax = plt.subplots(figsize=(8,6))
    ax.imshow(result_numba_hybrid, extent=[x_dim[0], x_dim[1], y_dim[0], y_dim[1]], cmap='inferno', origin='lower', aspect='equal')
    ax.set_title(' Serial Mandelbrot')
    ax.set_xlabel('Re(c)')
    ax.set_ylabel('Im(c)')
    plt.show()

    





    
    
