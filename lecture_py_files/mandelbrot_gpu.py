import pyopencl as cl
import numpy as np
import time
import matplotlib.pyplot as plt
from pathlib import Path

def mandelbrot_gpu_f32(N: int,
                   max_iter: int,
                   x_dim: tuple[float, float],
                   y_dim: tuple[float, float]):

    KERNEL_SRC = """
    __kernel void mandelbrot(
        __global int *result,
        const float x_min, const float x_max,
        const float y_min, const float y_max,
        const int N, const int max_iter)
    {
        // body to fill in
        int col = get_global_id(0);
        int row = get_global_id(1);
        if (col >= N || row >= N) return;   // guard against over-launch

        float c_real = x_min + col * (x_max - x_min) / (float)N;
        float c_imag = y_min + row * (y_max - y_min) / (float)N;

        float zr = 0.0f, zi = 0.0f;
        int count = 0;
        while (count < max_iter && zr*zr + zi*zi <=4.0f) {
            float tmp = zr*zr - zi*zi + c_real;
            zi = 2.0f * zr * zi + c_imag;
            zr = tmp;
            count++; 
        }
        result[row * N + col] = count;
    }
    """

    ## PyOpenCl setup
    ctx   = cl.create_some_context(interactive=False)
    queue = cl.CommandQueue(ctx)
    prog  = cl.Program(ctx, KERNEL_SRC).build()

    ## Mandelbrot
    image = np.zeros((N,N), dtype=np.int32)
    image_dev = cl.Buffer(ctx, cl.mem_flags.WRITE_ONLY, image.nbytes)

    ## Timing and save image
    # --- Warm up (first launch triggers a kernel compute) ---
    prog.mandelbrot(queue, (64, 64), None, image_dev,
                    np.float32(x_dim[0]), np.float32(x_dim[1]),
                    np.float32(y_dim[0]), np.float32(y_dim[1]),
                    np.int32(64), np.int32(max_iter))
    queue.finish()

    # --- Time the real run ---
    t0 = time.perf_counter()
    prog.mandelbrot(queue, (N, N), None, image_dev,
                    np.float32(x_dim[0]), np.float32(x_dim[1]),
                    np.float32(y_dim[0]), np.float32(y_dim[1]),
                    np.int32(N), np.int32(max_iter))
    queue.finish()
    elapsed = time.perf_counter() - t0

    cl.enqueue_copy(queue, image, image_dev)
    queue.finish()

    return elapsed, image

def mandelbrot_gpu_f64(N: int,
                   max_iter: int,
                   x_dim: tuple[float, float],
                   y_dim: tuple[float, float]):

    KERNEL_SRC = """
    #pragma OPENCL EXTENSION cl_khr_fp64 : enable
    __kernel void mandelbrot(
        __global int *result,
        const double x_min, const double x_max,
        const double y_min, const double y_max,
        const int N, const int max_iter)
    {
        // body to fill in
        int col = get_global_id(0);
        int row = get_global_id(1);
        if (col >= N || row >= N) return;   // guard against over-launch

        double c_real = x_min + col * (x_max - x_min) / (double)N;
        double c_imag = y_min + row * (y_max - y_min) / (double)N;

        double zr = 0.0, zi = 0.0;
        int count = 0;
        while (count < max_iter && zr*zr + zi*zi <=4.0) {
            double tmp = zr*zr - zi*zi + c_real;
            zi = 2.0 * zr * zi + c_imag;
            zr = tmp;
            count++; 
        }
        result[row * N + col] = count;
    }
    """

    ## PyOpenCl setup
    ctx   = cl.create_some_context(interactive=False)
    queue = cl.CommandQueue(ctx)
    prog  = cl.Program(ctx, KERNEL_SRC).build()

    dev = ctx.devices[0]
    if 'cl_khr_fp64' not in dev.extensions:
        print('No native fp64 -- Apple Silicon: emulated, expect large slowdown')

    ## Mandelbrot
    image = np.zeros((N,N), dtype=np.int32)
    image_dev = cl.Buffer(ctx, cl.mem_flags.WRITE_ONLY, image.nbytes)

    ## Timing and save image
    # --- Warm up (first launch triggers a kernel compute) ---
    prog.mandelbrot(queue, (64, 64), None, image_dev,
                    np.float64(x_dim[0]), np.float64(x_dim[1]),
                    np.float64(y_dim[0]), np.float64(y_dim[1]),
                    np.int32(64), np.int32(max_iter))
    queue.finish()

    # --- Time the real run ---
    t0 = time.perf_counter()
    prog.mandelbrot(queue, (N, N), None, image_dev,
                    np.float64(x_dim[0]), np.float64(x_dim[1]),
                    np.float64(y_dim[0]), np.float64(y_dim[1]),
                    np.int32(N), np.int32(max_iter))
    queue.finish()
    elapsed = time.perf_counter() - t0

    cl.enqueue_copy(queue, image, image_dev)
    queue.finish()

    return elapsed, image

if __name__ == '__main__':
    N = 1024
    max_iter = 100
    x_dim = (-2.5, 1.0)
    y_dim = (-1.5, 1.5)

    elapsed_f32, image_f32 = mandelbrot_gpu_f32(N=N, max_iter=max_iter, x_dim=x_dim, y_dim=y_dim)

    print(f'GPU {N}x{N} for f32: {elapsed_f32:.4f} s\n')
    file_folder = Path(__file__).resolve().parent
    figure_folder = file_folder.parent / 'figures'
    out_folder = figure_folder / 'mandelbrot_gpu_f32.png'
    plt.imshow(image_f32, cmap='hot', origin='lower')
    plt.axis('off')
    plt.savefig(out_folder, dpi=150, bbox_inches='tight')

    elapsed_f64, image_f64 = mandelbrot_gpu_f64(N=N, max_iter=max_iter, x_dim=x_dim, y_dim=y_dim)

    print(f'GPU {N}x{N} for f64: {elapsed_f64:.4f} s\n')
    out_folder = figure_folder / 'mandelbrot_gpu_f64.png'
    plt.imshow(image_f64, cmap='hot', origin='lower')
    plt.axis('off')
    plt.savefig(out_folder, dpi=150, bbox_inches='tight')