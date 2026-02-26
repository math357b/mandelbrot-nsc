"""
Mandelbrot Set Generator
Author : [ Mathias Jørgensen ]
Course : Numerical Scientific Computing 2026
"""
import numpy as np
import matplotlib.pyplot as plt
import time, statistics

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

# Lecture 1 - Naive Implementation
def mandelbrot_point_naive(c):

    # Parameters
    z = 0
    max_iter = 100

    for n in range(max_iter):
        z = z**2 + c 
        if abs(z) > 2:
            return n
    return max_iter 

# Lecture 1 - Naive Implementation        
def compute_mandelbrot_naive(x_dim = tuple[float, float],
                             y_dim = tuple[float, float],
                             res_x = int,
                             res_y = int):
    
    # Pulling out variables from tuples
    x_min, x_max = x_dim
    y_min, y_max = y_dim

    # Create 1D arrays
    x = np.linspace(x_min, x_max, res_x)
    y = np.linspace(y_min, y_max, res_y)

    #create array for n
    all_n = np.zeros((res_x, res_y), dtype=int)  

    for i in range(res_x):
        for j in range(res_y):
            c = x[i] + 1j * y[j]
            all_n[i, j] = mandelbrot_point_naive(c)
    return all_n

# Lecture 2 - Numpy Implementation
def compute_mandelbrot_numpy(x_dim = tuple[float, float],
                                  y_dim = tuple[float, float],
                                  res_x = int,
                                  res_y = int):
    
    # Pulling out variables from tuples
    x_min, x_max = x_dim
    y_min, y_max = y_dim
    
    # Parameters
    iter = 100
    
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

    for _ in range(iter):
        mask = np.abs(Z) <= 2           # Boolean mask
        Z[mask] = Z[mask]**2 + C[mask]  # Update only unescaped points (z = z**2 + c)
        M[mask] += 1                    # Increment iteration count

    return M

if __name__ == "__main__":

    # Parameters
    iterations = 3
    x_dim = (-2, 1)
    y_dim = (-1.5, 1.5)
    resolution = (1024, 1024)

    """
    ## Problem Size Scaling (Milestone 4 - Lecture 3)
    res_list = [256, 512, 1024, 2048, 4096]
    time_plot = []

    for i in res_list:
        elapsed_vectorized_grid, _ = benchmark(compute_mandelbrot_vectorized, x_dim, y_dim, i, i)
        time_plot.append(elapsed_vectorized_grid)
        print(f'Compute runtime for resolution{[i]}. Computation took {elapsed_vectorized_grid}')

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
    # Benchmark and plots of naive approach
    t_naive, M_naive = benchmark(compute_mandelbrot_naive, x_dim, y_dim, resolution, n_runs=iterations)
    print(f'Computing naive approach took {t_naive} seconds')
    plt.imshow(M_naive, cmap='hot')
    plt.title('Mandelbrot Set Figure L1')
    plt.colorbar()
    plt.savefig("mandelbrot_naive.png")
    """

    """
    # Benchmark and plots of vectorized approach
    t_vectorized, M_vectorized = benchmark(compute_mandelbrot_numpy, x_dim, y_dim, resolution, n_runs=iterations)
    print(f'Computing naive approach took {t_vectorized} seconds')
    plt.imshow(M_vectorized, cmap='hot')
    plt.title('Mandelbrot Set Figure L2')
    plt.colorbar()
    plt.savefig("mandelbrot_vectorized.png")
    plt.show()
    plt.close()
    """


    

        




  
