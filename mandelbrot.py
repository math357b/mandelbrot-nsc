"""
Mandelbrot Set Generator
Author : [ Mathias Jørgensen ]
Course : Numerical Scientific Computing 2026
"""
import numpy as np
import matplotlib as plt


def mandelbrotPoint(c):

    # Parameters
    z = 0
    max_iter = 100

    for n in range(max_iter):
        z = z**2 + c 

        if abs(z) > 2:
            return n
    return max_iter

if __name__ == "__main__":
    c = 1 + 1j
    print(c)
    mandelbrotPoint(c)

