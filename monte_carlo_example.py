import math, random, time, statistics

def estimate_pi_serial(num_samples):
    inside_cirle = 0
    for _ in range(num_samples):
        x, y = random.random(), random.random() # get random number for x and y
        if x**2 + y**2 <= 1:
            inside_cirle += 1
    pi_serial = 4 * inside_cirle / num_samples

    return pi_serial

if __name__ == '__main__':
    num_samples = 10_000_000
    times = []
    for _ in range(3):
        t0 = time.perf_counter()
        pi_estimate = estimate_pi_serial(num_samples=num_samples)
        times.append(time.perf_counter() - t0)
    t_serial = statistics.median(times)
    print(f"pi estimate: {pi_estimate:.6f} (error: {abs(pi_estimate - math.pi)})")
    print(f"Serial time: {t_serial:.3f}")
