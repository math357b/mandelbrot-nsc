"""
Mandelbrot Set Generator
Author : [ Mathias Jørgensen ]
Course : Numerical Scientific Computing 2026
"""
import numpy as np
import matplotlib.pyplot as plt
import time
import statistics

def benchmark(func, *args, n_runs=3):
    '''Benchmark a function by measuring execution time over multiple runs and return function result.

    Parameters
    ----------
    func : callable
        The function to benchmark.
    *args : tuple
        Positional arguments passed to 'func'.
    n_runs : int, optional
        Number of times to execute the function (default is 3).

    Returns
    -------
    median_t : float
        Median execution time in seconds.
    result : Any
        The result returned by the last function call.
    '''

    times = []

    for _ in range(n_runs):
        t0 = time.perf_counter()
        result = func(*args)
        times.append(time.perf_counter() - t0)
    median_t = statistics.median(times)
    print(f'{func.__name__}: Median: {median_t:.4f}s'
          f'(min={min(times):.4f}, max={max(times):.4f})')
    return median_t, result

# Lecture 1 - Naive Implementation
def mandelbrot_point_naive(c, max_iter):
    '''Compute the escape iteration count for a point in the Mandelbrot set (naive approach).

    This function iteratively applies the recurrence relation
    z_{n+1} = z_n^2 + c starting from z_0 = 0, and returns the
    iteration at which the magnitude of z exceeds 2. If the point
    does not escape within 'max_iter' iterations, 'max_iter' is returned.

    Parameters
    ----------
    c : complex
        Complex number representing the point in the complex plane.
    max_iter : int
        Maximum number of iterations to test for divergence.

    Returns
    -------
    n : int
        Number of iterations before divergence (|z| > 2).
        Returns 'max_iter' if the point does not diverge.
    '''

    # Parameters
    z = 0

    for n in range(max_iter):
        z = z**2 + c 
        if abs(z) > 2:
            return n
    return max_iter 

# Lecture 1 - Naive Implementation        
def compute_mandelbrot_naive(x_dim: tuple[float, float],
                             y_dim: tuple[float, float],
                             res: tuple[int, int],
                             max_iter: int = 100):
    
    '''Compute a Mandelbrot set grid using a naive point-by-point approach.

    The function evaluates the escape iteration count for each point in a
    2D grid of the complex plane, using 'mandelbrot_point_naive'.

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
            all_n[i, j] = mandelbrot_point_naive(c, max_iter)
    return all_n

# Lecture 2 - Numpy Implementation
def compute_mandelbrot_numpy(x_dim: tuple[float, float],
                             y_dim: tuple[float, float],
                             res: tuple[int, int],
                             max_iter: int = 100):
    
    '''Compute a Mandelbrot set grid using a vectorized Numpy approach.

    The function evaluates the escape iteration count for each point in a
    2D grid of the complex plane using array operations instead of explicit
    Python loops.

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
    M : ndarray of shape (res_y, res_x)
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

    # Create 2D arrays
    X, Y = np.meshgrid(x, y)

    # Create complex grid
    C = X + 1j * Y

    # Initialize Z and M arrays
    Z = np.zeros(C.shape, dtype=complex) # same as #np.zeros_like(C)
    M = np.zeros(C.shape, dtype=int) 

    for _ in range(max_iter):
        mask = np.abs(Z) <= 2           # Boolean mask
        Z[mask] = Z[mask]**2 + C[mask]  # Update only unescaped points (z = z**2 + c)
        M[mask] += 1                    # Increment iteration count

    return M

# Lecture 2 - Numpy Implementation
def compute_row_sums(A = np.ndarray,
                     N = int):
    for i in range(N):
        s = np.sum(A[i,:])
    
    return s

# Lecture 2 - Numpy Implementation
def compute_column_sums(A = np.ndarray,
                        N = int):
    for j in range(N):
        s = np.sum(A[:,j])
    
    return s

if __name__ == "__main__":

    # Parameters
    n_runs = 3
    x_dim = (-2, 1)
    y_dim = (-1.5, 1.5)
    resolution = (512, 512)

    """
    ## Problem Size Scaling (Milestone 4 - Lecture 3)
    res_list = [256, 512, 1024, 2048, 4096]
    time_plot = []

    for i in res_list:
        elapsed_numpy_grid, _ = benchmark(compute_mandelbrot_numpy, x_dim, y_dim, i, i)
        time_plot.append(elapsed_numpy_grid)
        print(f'Compute runtime for resolution{[i]}. Computation took {elapsed_numpy_grid}')

    plt.figure()
    plt.plot(res_list, time_plot)
    plt.title("Grid size vs runtime")
    plt.xlabel("Grid size")
    plt.ylabel("Runtime")
    plt.grid(True)
    plt.savefig("gridsize_vs_runtime")
    plt.show()
    """

    """
    ## Memory Access Patterns (Milestone 3 - Lecture 2)
    N = 10000
    A = np.random.rand(N,N)
    t_row_sum, _ = benchmark(compute_row_sums, A, N)
    t_column_sum, _ = benchmark(compute_column_sums, A, N)

    # results for row and column respectively:
    # Median: 0.1343s(min=0.1242, max=0.1505)
    # Median: 1.0571s(min=1.0505, max=1.0861)

    A_f = np.asfortranarray(A)
    t_row_sum, _ = benchmark(compute_row_sums, A_f, N)
    t_column_sum, _ = benchmark(compute_column_sums, A_f, N)
    
    # results for row and column respectively:
    # Median = 1.1206s(min=1.0590, max=1.1505)
    # Median = 0.1336s(min=0.1300, max=0.1393)
    """

    # Benchmark and plots of naive approach
    t_naive, M_naive = benchmark(compute_mandelbrot_naive, x_dim, y_dim, resolution, n_runs=n_runs)
    print(f'Computing naive approach took {t_naive} seconds')
    plt.imshow(M_naive, cmap='hot', origin='lower')
    plt.title('Mandelbrot Set Figure L1')
    plt.colorbar()
    plt.show()
    #plt.savefig("mandelbrot_naive.png")

    """
    # Benchmark and plots of numpy approach
    t_numpy, M_numpy = benchmark(compute_mandelbrot_numpy, x_dim, y_dim, resolution, n_runs=iterations)
    print(f'Computing naive approach took {t_numpy} seconds')
    plt.imshow(M_numpy, cmap='hot')
    plt.title('Mandelbrot Set Figure L2')
    plt.colorbar()
    plt.savefig("mandelbrot_numpy.png")
    plt.show()
    plt.close()
    """


    

        




  
