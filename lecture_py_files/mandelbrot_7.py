from dask import delayed
from dask.distributed import Client, LocalCluster
import dask, time, statistics
import numpy as np
import matplotlib.pyplot as plt
from lecture_py_files.mandelbrot_3 import compute_mandelbrot_full
from lecture_py_files.mandelbrot_5 import mandelbrot_chunk

def mandelbrot_dask(N: int,
                    x_dim: tuple[float, float],
                    y_dim: tuple[float, float],
                    max_iter: int = 100,
                    n_chunks: int = 32):
    
    chunk_size = max(1, N // n_chunks)
    tasks = []
    row = 0

    while row < N:
        row_end = min(row + chunk_size, N)
        tasks.append(delayed(mandelbrot_chunk)(row, row_end, N, x_dim, y_dim, max_iter))
        row = row_end
    
    parts = dask.compute(*tasks)

    return np.vstack(parts)

def experiment_1():
    # Parameters
    resolutions = [1024, 2048, 4096, 8192]
    iterations = 100
    n_runs = 3
    x_dim = (-2.5, 1.0)
    y_dim = (-1.25, 1.25)
    chunk_values = [1, 2, 4, 8, 16, 32, 64, 128]

    client = Client("tcp://10.92.1.30:8786")

    n_workers = len(client.scheduler_info()['workers'])
    threads_per_worker = [w['nthreads'] for w in client.scheduler_info()['workers'].values()]

    print(f"Number of workers: {n_workers}")
    print(f"Threads per worker: {threads_per_worker}")
    
    # Warmup
    client.run(lambda: mandelbrot_chunk(row_start=0, row_end=8, N=8, x_dim=x_dim, y_dim=y_dim, max_iter=iterations))
    _ = compute_mandelbrot_full(x_dim=x_dim, y_dim=y_dim, res=(64,64))  # Numba warmup

    # Store speedups vs resolution
    speedups = []

    # Compute time for Numba and Dask distributed for different resolutions
    for N in resolutions:
        res = (N, N)

        # Numba serial baseline
        numba_times = []
        for _ in range(n_runs):
            t0 = time.perf_counter()
            compute_mandelbrot_full(x_dim=x_dim, y_dim=y_dim, res=res, max_iter=iterations)
            numba_times.append(time.perf_counter() - t0)
        t_numba = statistics.median(numba_times)

        # Dask distributed: sweep chunk sizes
        wall_times = []
        for n_chunks in chunk_values:
            dask_times = []
            for _ in range(n_runs):
                t0 = time.perf_counter()
                mandelbrot_dask(N, x_dim, y_dim, max_iter=iterations, n_chunks=n_chunks)
                dask_times.append(time.perf_counter() - t0)
            median_time = statistics.median(dask_times)
            wall_times.append((n_chunks, median_time))

        # Determine best chunk size
        wall_times_array = np.array(wall_times)
        optimal_idx = np.argmin(wall_times_array[:,1])
        n_optimal = int(wall_times_array[optimal_idx,0])
        t_dask_best = wall_times_array[optimal_idx,1]

        # Overall speedup
        speedup = t_numba / t_dask_best
        speedups.append((N, t_numba, t_dask_best, speedup))

        # Print results
        print(f"\nResolution N={N}")
        print(f"Numba serial: {t_numba:.2f}s")
        print(f"Dask best chunk={n_optimal}: {t_dask_best:.2f}s | Speedup={speedup:.2f}x")
        print("Wall time vs chunks:")
        for nc, wt in wall_times:
            print(f"Chunks: {nc:3d} | Time: {wt:.2f}s")

        # Plot wall time vs chunk size
        plt.figure()
        plt.plot(wall_times_array[:,0], wall_times_array[:,1], marker='o')
        plt.xscale('log', base=2)
        plt.xlabel("Number of chunks")
        plt.ylabel("Wall time (s)")
        plt.title(f"Wall time vs chunk count (N={N})")
        plt.grid(True)
        plt.savefig(f"mandelbrot_walltime_N{N}.png")
        plt.close()

    client.close()

    # Plot speedup vs resolution
    speedups_array = np.array(speedups)
    plt.figure()
    plt.plot(speedups_array[:,0], speedups_array[:,3], marker='o')
    plt.xlabel("Resolution N")
    plt.ylabel("Speedup (Numba / Dask)")
    plt.title("Dask distributed speedup over Numba serial")
    plt.grid(True)
    plt.savefig("mandelbrot_speedup_vs_resolution.png")
    plt.close()

def experiment_2():
    # Parameters
    resolutions = [1024, 2048, 4096, 8192]
    iterations = 100
    n_runs = 3
    x_dim = (-2.5, 1.0)
    y_dim = (-1.25, 1.25)
    optimal_chunks = {
    1024: 2,
    2048: 2,
    4096: 2,
    8192: 4}

    client = Client("tcp://10.92.1.30:8786")

    n_workers = len(client.scheduler_info()['workers'])
    threads_per_worker = [w['nthreads'] for w in client.scheduler_info()['workers'].values()]

    print(f"Number of workers: {n_workers}")
    print(f"Threads per worker: {threads_per_worker}")
    
    # Warmup
    client.run(lambda: mandelbrot_chunk(row_start=0, row_end=8, N=8, x_dim=x_dim, y_dim=y_dim, max_iter=iterations))
    _ = compute_mandelbrot_full(x_dim=x_dim, y_dim=y_dim, res=(64,64))  # Numba warmup

    # Compute time for Numba and Dask distributed for different resolutions
    for N in resolutions:
        res = (N, N)

        # Numba serial baseline
        numba_times = []
        for _ in range(n_runs):
            t0 = time.perf_counter()
            compute_mandelbrot_full(x_dim=x_dim, y_dim=y_dim, res=res, max_iter=iterations)
            numba_times.append(time.perf_counter() - t0)
        t_numba = statistics.median(numba_times)

        # Dask distributed: sweep chunk sizes
        dask_times = []
        n_chunks = optimal_chunks[N]

        for _ in range(n_runs):
            t0 = time.perf_counter()
            mandelbrot_dask(N, x_dim, y_dim, max_iter=iterations, n_chunks=n_chunks)
            dask_times.append(time.perf_counter() - t0)
        t_dask = statistics.median(dask_times)

        # Speedup
        speedup = t_numba / t_dask

        # Print results
        print(f"\nResolution N={N}")
        print(f"Optimal n_chunk: {n_chunks}")
        print(f"Numba time: {t_numba:.2f}s")
        print(f"Dask time:  {t_dask:.2f}s")
        print(f"Speedup:    {speedup:.2f}x")

    client.close()

if __name__ == '__main__':  
    #experiment_1()
    experiment_2()
