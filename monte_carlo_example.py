from multiprocessing import Pool
import os
import math, random, time, statistics

def estimate_pi_serial(num_samples: int):
    inside_cirle = 0
    for _ in range(num_samples):
        x, y = random.random(), random.random() # get random number for x and y
        if x**2 + y**2 <= 1:
            inside_cirle += 1
    pi_serial = 4 * inside_cirle / num_samples
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
    samples_per_process = num_samples // num_processes
    tasks = [samples_per_process] * num_processes
    with Pool(processes=num_processes) as pool:
        results = pool.map(estimate_pi_chunk, tasks)
    return 4 * sum(results) / num_samples

if __name__ == '__main__':
    num_samples = 10_000_000
    for num_proc in range(1, os.cpu_count() + 1):
        times_serial = []
        times_parallel = []
        for _ in range(3):
            t0_serial = time.perf_counter()
            pi_est_serial = estimate_pi_serial(num_samples=num_samples)
            times_serial.append(time.perf_counter() - t0_serial)
            t0_parallel = time.perf_counter()
            pi_est_parallel = estimate_pi_parallel(num_samples=num_samples, num_processes=num_proc)
            times_parallel.append(time.perf_counter() - t0_parallel)
        t_serial = statistics.median(times_serial)
        t_parallel = statistics.median(times_parallel)

        speedup = t_serial/t_parallel
        efficiency = speedup/num_proc

        print(f"workers: {num_proc} | time: {t_parallel:.3f}(s) | speedup: {speedup:.3f}x | efficiency: {efficiency:.3f}")
