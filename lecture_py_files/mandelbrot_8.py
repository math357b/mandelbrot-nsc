import numpy as np
import matplotlib.pyplot as plt

def mandelbrot_trajectory_divergence(N: int,
                                     x_dim: tuple[float, float],
                                     y_dim: tuple[float, float],
                                     max_iter: int,
                                     tau: float,
                                     plot: bool = False):
    
    x = np.linspace(start=x_dim[0], stop=x_dim[1], num=N)
    y = np.linspace(start=y_dim[0], stop=y_dim[1], num=N)

    C64 = (x[np.newaxis, :] + 1j * y[:, np.newaxis]).astype(np.complex128)
    C32 = C64.astype(np.complex64)
    z32 = np.zeros_like(C32)
    z64 = np.zeros_like(C64)
    diverge = np.full((N, N), max_iter, dtype=np.int32)
    active = np.ones((N, N), dtype=bool)

    for k in range(max_iter):
        if not active.any(): break
        z32[active] = z32[active]**2 + C32[active]
        z64[active] = z64[active]**2 + C64[active]
        diff = (np.abs(z32.real.astype(np.float64) - z64.real) + np.abs(z32.imag.astype(np.float64) - z64.imag))
        newly = active & (diff > tau)
        diverge[newly] = k
        active[newly] = False

    if plot == True:
        plt.imshow(diverge, cmap='plasma', origin='lower', extent=[x_dim[0], x_dim[1], y_dim[0], y_dim[1]])
        plt.colorbar(label='First divergence iteration')
        plt.title(f'Trajectory divergence (tau={tau})')
        plt.show()

    return diverge

if __name__ == '__main__':




