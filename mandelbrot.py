"""
Mandelbrot Set Generator
Author : [ Mathias Jørgensen ]
Course : Numerical Scientific Computing 2026
"""
import numpy as np
import matplotlib.pyplot as plt
import time

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
    
    x = np.linspace(x_min, x_max, resx)
    y = np.linspace(y_min, y_max, resy)

    #create array for n
    all_n = np.zeros((resx, resy), dtype=int)  

    for i in range(resx):
        for j in range(resy):
            c = x[i] + 1j * y[j]
            all_n[i, j] = mandelbrot_point(c)
    return all_n

if __name__ == "__main__":
    start = time.time()
    all_n = compute_mandelbrot(-2, 1, -1.5, 1.5, 4092, 4092)
    elapsed = time.time() - start
    print(f'It took {elapsed} seconds')
    
    plt.imshow(all_n)
    plt.colorbar()
    plt.show()
