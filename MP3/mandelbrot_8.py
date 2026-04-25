import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

def mandelbrot_trajectory_divergence(N: int,
                                     x_dim: tuple[float, float],
                                     y_dim: tuple[float, float],
                                     max_iter: int,
                                     tau: float):
    '''Compute a Mandelbrot divergence map using double-precision trajectory
    comparison.

    This function evolves two parallel Mandelbrot iterations per point:
    one in float32 precision and one in float64 precision. Divergence is
    detected by measuring the deviation between the two trajectories. When
    the difference exceeds a threshold 'tau', the point is marked as diverged.

    Parameters
    ----------
    N : int
        Resolution of the grid (produces an N x N output).
    x_dim : tuple of float
        (x_min, x_max) range of the real axis.
    y_dim : tuple of float
        (y_min, y_max) range of the imaginary axis.
    max_iter : int
        Maximum number of iterations for the Mandelbrot recurrence.
    tau : float
        Divergence threshold between float32 and float64 trajectories.

    Returns
    -------
    diverge : ndarray of shape (N, N)
        Array where each entry contains the iteration index at which the
        trajectory difference exceeded 'tau'. Points that never diverge
        within 'max_iter' are assigned 'max_iter'.
    '''
    
    x = np.linspace(start=x_dim[0], stop=x_dim[1], num=N)
    y = np.linspace(start=y_dim[0], stop=y_dim[1], num=N)

    C64 = (x[np.newaxis, :] + 1j * y[:, np.newaxis]).astype(np.complex128)
    C32 = C64.astype(np.complex64)
    z32 = np.zeros_like(C32)
    z64 = np.zeros_like(C64)
    diverge = np.full((N, N), max_iter, dtype=np.int32)
    active = np.ones((N, N), dtype=bool)

    for k in range(max_iter):
        if not active.any():
            break
        z32[active] = z32[active]**2 + C32[active]
        z64[active] = z64[active]**2 + C64[active]
        diff = (np.abs(z32.real.astype(np.float64) - z64.real) + np.abs(z32.imag.astype(np.float64) - z64.imag))
        newly = active & (diff > tau)
        diverge[newly] = k
        active[newly] = False

    return diverge

def mandelbrot_escape_count(N: int,
                            x_dim: tuple[float, float],
                            y_dim: tuple[float, float],
                            max_iter: int,
                            escape_radius: float = 2.0):
    
    '''Compute the Mandelbrot escape-time fractal using vectorized NumPy arrays.

    This function iterates the recurrence relation z_{n+1} = z_n^2 + c
    over a 2D grid of complex values and records the iteration at which
    each point escapes a given radius.

    Parameters
    ----------
    N : int
        Resolution of the grid (produces an N x N output).
    x_dim : tuple of float
        (x_min, x_max) range of the real axis.
    y_dim : tuple of float
        (y_min, y_max) range of the imaginary axis.
    max_iter : int
        Maximum number of iterations to compute for each point.
    escape_radius : float, optional
        Radius beyond which a point is considered to have escaped
        (default is 2.0).

    Returns
    -------
    escape : ndarray of shape (N, N)
        2D array where each element contains the iteration index at which
        the corresponding point escaped. Points that do not escape within
        'max_iter' iterations are assigned 'max_iter'.
    '''

    x = np.linspace(x_dim[0], x_dim[1], N)
    y = np.linspace(y_dim[0], y_dim[1], N)

    C = (x[np.newaxis, :] + 1j * y[:, np.newaxis])
    z = np.zeros_like(C)
    escape = np.full((N,N), max_iter, dtype=np.int32)
    active = np.ones((N,N), dtype=bool)

    for k in range(max_iter):
        if not active.any():
            break
        z[active] = z[active]**2 + C[active]

        escaped = active & (np.abs(z) > escape_radius)
        escape[escaped] = k
        active[escaped] = False

    return escape

def mandelbrot_sensitivity_map(N: int,
                               max_iter: int,
                               x_dim: tuple[float, float],
                               y_dim: tuple[float, float]):
    
    '''Compute a sensitivity map of the Mandelbrot set by comparing escape times under small deviations.

    This function evaluates how sensitive the Mandelbrot escape-time
    function is to deviations in the input complex plane.

    Parameters
    ----------
    N : int
        Resolution of the grid (produces an N x N output).
    max_iter : int
        Maximum number of iterations for escape-time computation.
    x_dim : tuple of float
        (x_min, x_max) range of the real axis.
    y_dim : tuple of float
        (y_min, y_max) range of the imaginary axis.

    Returns
    -------
    kappa : ndarray of shape (N, N)
        Sensitivity measure defined as:
            |n_base - n_perturb| / (eps_32 * n_base)
    cmap_k : matplotlib.colors.Colormap
        Colormap used for visualizing the sensitivity map.
        Invalid values (NaN) are masked.
    vmax : float
        99th percentile of kappa, useful for visualization scaling.
    '''
    
    x = np.linspace(x_dim[0], x_dim[1], N)
    y = np.linspace(y_dim[0], y_dim[1], N)

    C = (x[np.newaxis, :] +  1j * y[:, np.newaxis]).astype(np.complex128)
    eps32 = float(np.finfo(np.float32).eps)
    delta = np.maximum(eps32 / np.abs(C), 1e-10)

    def escape_count(C, max_iter):
        z = np.zeros_like(C)
        cnt = np.full(C.shape, max_iter, dtype=np.int32)
        esc = np.zeros(C.shape, dtype=bool)
        for k in range(max_iter):
            z[~esc] = z[~esc]**2 + C[~esc]
            newly = ~esc & (np.abs(z) > 2.0)
            cnt[newly] = k
            esc[newly] = True
        return cnt
    
    n_base = escape_count(C=C, max_iter=max_iter).astype(float)
    n_perturb = escape_count(C=C+delta, max_iter=max_iter).astype(float)
    dn = np.abs(n_base - n_perturb)
    kappa = np.where(n_base > 0, dn / (eps32 * n_base), np.nan)
    cmap_k = plt.cm.hot.copy()
    cmap_k.set_bad('0.25')
    
    vmax = np.nanpercentile(kappa, 99)

    return kappa, cmap_k, vmax

if __name__ == '__main__':

    N = 512
    #x_dim = (-0.7530, -0.7490)
    #y_dim = (0.0990, 0.1030)
    x_dim = (-2.5, 1.0)
    y_dim = (-1.5, 1.5)

    max_iter = 1000
    tau = 0.01

    mandelbrot_sensitivity_map(N, max_iter, x_dim, y_dim)
    exit()

    diverge = mandelbrot_trajectory_divergence(N=N, x_dim=x_dim, y_dim=y_dim, max_iter=max_iter, tau=tau)
    escape = mandelbrot_escape_count(N=N, x_dim=x_dim, y_dim=y_dim, max_iter=max_iter)

    plt.figure()
    plt.subplot(1, 2, 1)
    plt.imshow(diverge, cmap='plasma', origin='lower', extent=[x_dim[0], x_dim[1], y_dim[0], y_dim[1]])
    plt.colorbar(label='First divergence iteration')
    plt.title(f'Trajectory divergence (tau={tau})')
    plt.subplot(1, 2, 2)
    plt.imshow(escape, cmap='viridis', origin='lower', extent=[x_dim[0], x_dim[1], y_dim[0], y_dim[1]])
    plt.colorbar(label='Escape iteration')
    plt.title('Escape iteration count')
    plt.close()

    kappa, cmap_k, vmax = mandelbrot_sensitivity_map(N=N, max_iter=max_iter, x_dim=x_dim, y_dim=y_dim)

    plt.figure()
    plt.imshow(kappa, cmap=cmap_k, origin='lower',
            extent=[x_dim[0], x_dim[1], y_dim[0], y_dim[1]],
            norm=LogNorm(vmin=1, vmax=vmax))
    plt.colorbar(label=r'$\kappa(c)$ (log scale, $\kappa \geq 1$)')
    plt.title(r'Condition number approx $\kappa(c) = |\Delta n|\,/\,(\varepsilon_{32}\,n(c))$')
    plt.show()

