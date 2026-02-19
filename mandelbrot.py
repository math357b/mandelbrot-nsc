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

def mandelbrot_point(c):

    # Parameters
    z = 0
    max_iter = 100

    for n in range(max_iter):
        z = z**2 + c 
        if abs(z) > 2:
            return n
    return max_iter 
        
def compute_mandelbrot(x_min, x_max, y_min, y_max, resx, resy):
    
    # Create 1D arrays
    x = np.linspace(x_min, x_max, resx)
    y = np.linspace(y_min, y_max, resy)

    # Create 2D arrays
    X, Y = np.meshgrid(x, y)

    # Create complex grid
    C = X + 1j * Y

    print(f'Shape: {C.shape}') # (1024, 1024)
    print(f'Type: {C.dtype}')  # complex128

    #create array for n
    all_n = np.zeros((resx, resy), dtype=int)  

    for i in range(resx):
        for j in range(resy):
            c = x[i] + 1j * y[j]
            all_n[i, j] = mandelbrot_point(c)
    return all_n

if __name__ == "__main__":
    """
    start = time.time()
    all_n = compute_mandelbrot(-2, 1, -1.5, 1.5, 1024, 1024)
    elapsed = time.time() - start
    print(f'Computation took {elapsed:.2f} seconds')
    """
    iterations = 3

    t, M = benchmark(compute_mandelbrot, -2, 1, -1.5, 1.5, 1024, 1024, n_runs=iterations)
    
    all_n = M

    plt.imshow(all_n, cmap='hot')
    plt.title('Mandelbrot Set Figure L1')
    plt.colorbar()

    plt.savefig("mandelbrot_naive.png")
    plt.show()
    plt.close()
