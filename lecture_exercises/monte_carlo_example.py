from multiprocessing import Pool
import os
import math, random, time, statistics
import matplotlib.pyplot as plt

def estimate_pi_serial(num_samples: int):
    inside_circle = estimate_pi_chunk(num_samples=num_samples)
    pi_serial = 4 * inside_circle / num_samples
    return pi_serial

def estimate_pi_chunk(num_samples: int):
    inside_cirle = 0
    for _ in range(num_samples):
        x, y = random.random(), random.random() # get random number for x and y
        if x**2 + y**2 <= 1:
            inside_cirle += 1
    return inside_cirle

def estimate_pi_parallel(num_samples: int,
                         num_processes: int = 4):
    
    if num_processes == 1:
        return estimate_pi_serial(num_samples=num_samples)
    
    samples_per_process = num_samples // num_processes
    tasks = [samples_per_process] * num_processes
    with Pool(processes=num_processes) as pool:
        results = pool.map(estimate_pi_chunk, tasks)
    return 4 * sum(results) / num_samples

def plot_worker_speedup(title: str,
                        workers: int,
                        speedup: float):
    plt.figure()
    plt.plot(workers, speedup, marker='o')
    plt.title(f"{title}: Worker vs. Speedup")
    plt.xlabel("Number of worker processes")
    plt.ylabel("Speedup (relative to serial)")
    plt.show()

if __name__ == '__main__':
    num_samples = 10_000_000
    workers_list = []
    speedups = []

    # Measure serial time once
    times_serial = []
    for _ in range(3):
        t0_serial = time.perf_counter()
        pi_est_serial = estimate_pi_serial(num_samples=num_samples)
        times_serial.append(time.perf_counter() - t0_serial)
    t_serial = statistics.median(times_serial)

    for num_proc in range(1, os.cpu_count() + 1):
        times_parallel = []
        for _ in range(3):
            t0_parallel = time.perf_counter()
            pi_est_parallel = estimate_pi_parallel(num_samples=num_samples,
                                                   num_processes=num_proc)
            times_parallel.append(time.perf_counter() - t0_parallel)

        t_parallel = statistics.median(times_parallel)

        speedup = t_serial / t_parallel
        efficiency = speedup / num_proc

        workers_list.append(num_proc)
        speedups.append(speedup)

        print(f"workers: {num_proc} | "
              f"time(parallel): {t_parallel:.3f}(s) | "
              f"time(serial): {t_serial:.3f}(s)| "
              f"speedup: {speedup:.3f}x | "
              f"efficiency: {efficiency:.3f}")
        
    plot_worker_speedup(title="Monte Carlo", workers=workers_list, speedup=speedups)
    plt.savefig("figures/Monte_Carlo_worker_speedup.png")