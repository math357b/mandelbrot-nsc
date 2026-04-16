from multiprocessing import Pool
import random, time, os
from functools import reduce

def monte_carlo_chunk(num_samples):
    """Estimate pi contributions for num_samples random points."""
    inside = 0
    for _ in range(num_samples):
        x, y = random.random(), random.random()
        if x*x + y*y <= 1:
            inside += 1
    return inside

def test_granularity(total_work, chunk_size, n_proc):
    n_chunks = total_work // chunk_size
    tasks = [chunk_size] * n_chunks
    start_time = time.perf_counter()
    if n_proc == 1:
        results = [monte_carlo_chunk(s) for s in tasks]
    else:
        with Pool(processes=n_proc) as pool:
            results = pool.map(monte_carlo_chunk, tasks)

    end_time = time.perf_counter() - start_time
    results = 4 * sum(results) / total_work

    return end_time, results

def subtract_seven(x):
    return x - 7

if __name__ == '__main__':
    
    # E1 - Lecture 5
    total_work = 1_000_000
    n_proc = os.cpu_count() // 2
    chunk_sizes = [10, 100, 1_000, 10_000, 100_000, 1_000_000] 
    print(f"{'L':>12} | {'serial (s)':>12} | {'parallel (s)':>12}")
    for L in chunk_sizes:
        t_serial, _ = test_granularity(total_work=total_work, chunk_size=L, n_proc=1)
        t_parallel, pi = test_granularity(total_work=total_work, chunk_size=L, n_proc=n_proc)
        print(f'{L:12d} | {t_serial:12.4f} | {t_parallel:12.4f}  pi={pi:.4f}')

    # E2 - Lecture 5
    N = 1_000_000
    data = [random.randint(10, 100) for _ in range(N)]

    # Part 1 - serial pipeline (map / filter / reduced chained)
    t0 = time.perf_counter()
    result_serial = reduce(lambda a, b: a + b,
                           filter(lambda x: x % 2 == 1,
                                  map(subtract_seven, data)))
    t_serial = time.perf_counter() - t0

    # Part 2 - replace map() with Pool.map
    t0 = time.perf_counter()
    with Pool() as pool:
        mapped = pool.map(subtract_seven, data)
    result_parallel = reduce(lambda a, b: a + b,
                           filter(lambda x: x % 2 == 1, mapped))
    t_parallel = time.perf_counter() - t0

    print(f'Serial:   {t_serial:.4f}s | Result={result_serial}')
    print(f'Parallel: {t_parallel:.4f}s | Results={result_parallel}')
    print(f'Speedup:  {t_serial / t_parallel:.4f}x')
    