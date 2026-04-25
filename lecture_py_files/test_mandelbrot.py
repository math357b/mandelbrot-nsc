import pytest
import numpy as np
import dask
from dask import delayed
import time
import statistics

# All imports from lectures
from lecture_py_files.mandelbrot_1_2 import mandelbrot_point_naive, compute_mandelbrot_naive, compute_mandelbrot_numpy
from lecture_py_files.mandelbrot_3 import mandelbrot_point_numba, compute_mandelbrot_hybrid
from lecture_py_files.mandelbrot_5 import mandelbrot_pixel, mandelbrot_chunk, mandelbrot_serial, mandelbrot_parallel
from lecture_py_files.mandelbrot_7 import mandelbrot_dask
from lecture_py_files.mandelbrot_gpu import mandelbrot_gpu_f32, mandelbrot_gpu_f64

#----------------------------
# Test Numba
#----------------------------

# Test 1: Equal Point Functions
POINT_IMPLEMENTATIONS = [mandelbrot_point_naive, mandelbrot_point_numba]

KNOWN_POINT_CASES = [
    (0+0j, 100, 100),   # origin: never escapes
    (5.0+0j, 100, 1),   # far outside, escapes on iteration 1
    (-2.5+0j, 100, 1),  # left tip of set
]

@pytest.mark.parametrize('point_func', POINT_IMPLEMENTATIONS)
@pytest.mark.parametrize('c, max_iter, expected', KNOWN_POINT_CASES)
def test_numba_naive_point_matches(point_func, c, max_iter, expected):
    result = point_func(c, max_iter)
    assert result == expected

# Test 2: Verify grid functions produce same results
GRID_TEST_CASES = [
    ((-2.0, 1.0), (-1.5, 1.5), (1024, 1024), 100)
]

@pytest.mark.parametrize('x_dim, y_dim, res, max_iter', GRID_TEST_CASES)
def test_naive_and_numba_grid_match(x_dim, y_dim, res, max_iter):
    result_naive = compute_mandelbrot_naive(x_dim=x_dim, y_dim=y_dim, res=res, max_iter=max_iter)
    result_numba = compute_mandelbrot_hybrid(x_dim=x_dim, y_dim=y_dim, res=res, max_iter=max_iter)
    
    np.testing.assert_array_equal(result_naive, result_numba)

#----------------------------
# Test Multiprocessing pool   
#----------------------------

# Test 1: Test the worker function (mandelbrot pixel) in isolation — it is pure
MULTIPROCESSING_IMPLEMENTATIONS = [mandelbrot_pixel]

PIXEL_TEST_CASES = [
    (0.0, 0.0, 100, 100),   # origin: never escapes
    (5.0, 0.0, 100, 1),     # far outside, escapes on iteration 1
    (-2.5, 0.0, 100, 1),    # left tip of set
]

@pytest.mark.parametrize('worker_func', MULTIPROCESSING_IMPLEMENTATIONS)
@pytest.mark.parametrize('c_real, c_imag, max_iter, expected', PIXEL_TEST_CASES)
def test_mandelbrot_pixel(worker_func, c_real, c_imag, max_iter, expected):
    result = worker_func(c_real, c_imag, max_iter)
    assert result == expected

# Test 2: Integration test: the assembled grid must match the serial result on a small grid
ASSEMBLED_SERIAL_TEST = [
    (32, (-2.0, 1.0), (-1.5, 1.5), 100, 4, 4),
    (32, (-2.0, 1.0), (-1.5, 1.5), 1000, 2, 32)
]

@pytest.mark.parametrize('N, x_dim, y_dim, max_iter, num_workers, n_chunks', ASSEMBLED_SERIAL_TEST)
def test_pixel_match_serial_small_grid(N, x_dim, y_dim, max_iter, num_workers, n_chunks):
    result_serial = mandelbrot_serial(N=N, x_dim=x_dim, y_dim=y_dim, max_iter=max_iter)
    result_parallel = mandelbrot_parallel(N=N, x_dim=x_dim, y_dim=y_dim, max_iter=max_iter, num_workers=num_workers, n_chunks=n_chunks)

    np.testing.assert_array_equal(result_serial, result_parallel)

#----------------------------
# Test Dask
#----------------------------

# Test 1: Test the underlying compute function in dask matches the serial result on a small grid
DASK_SERIAL_TEST = [
    (32, (-2.0, 1.0), (-1.5, 1.5), 100, 32)
]

@pytest.mark.parametrize('N, x_dim, y_dim, max_iter, n_chunks', DASK_SERIAL_TEST)
def test_dask_match_serial_small_grid(N, x_dim, y_dim, max_iter, n_chunks):
    result_serial = mandelbrot_serial(N=N, x_dim=x_dim, y_dim=y_dim, max_iter=max_iter)
    result_dask = mandelbrot_dask(N=N, x_dim=x_dim, y_dim=y_dim, max_iter=max_iter, n_chunks=n_chunks)

    np.testing.assert_array_equal(result_serial, result_dask)

# Test 2: Integration: future = client.submit(f, arg); assert client.gather(future) == expected
CHUNK_DELAYED_TEST = [
    (0, 4, 8, (-2.0, 1.0), (-1.5, 1.5), 100),
    (0, 1024, 1024, (-2.0, 1.0), (-1.5, 1.5), 100),
    (0, 1024, 1024, (-2.0, 1.0), (-1.5, 1.5), 1000)
]

@pytest.mark.parametrize('row_start, row_end, N, x_dim, y_dim, max_iter', CHUNK_DELAYED_TEST)
def test_single_chunk_dask_delayed_match(row_start, row_end, N, x_dim, y_dim, max_iter):

    result_expected = mandelbrot_chunk(row_start=row_start,
                                row_end=row_end,
                                N=N,
                                x_dim=x_dim,
                                y_dim=y_dim,
                                max_iter=max_iter)
    
    result_future = delayed(mandelbrot_chunk)(row_start=row_start,
                                       row_end=row_end,
                                       N=N,
                                       x_dim=x_dim,
                                       y_dim=y_dim,
                                       max_iter=max_iter)

    result_actual = dask.compute(result_future)[0]

    np.testing.assert_array_equal(result_actual, result_expected)

#----------------------------
# Test Perfomance Benchmarks
#----------------------------

# Test 1: Performance regression tests use relative assertions on a small grid: 
# assert numpy time < 0.1 * naive time (10x speedup)
NAIVE_NUMPY_TEST = [
    ((-2.0, 1.0), (-1.5, 1.5), (512, 512), 100)
]

@pytest.mark.parametrize('x_dim, y_dim, res, max_iter', NAIVE_NUMPY_TEST)
def test_numpy_naive_time(x_dim, y_dim, res, max_iter):

    naive_times = []
    numpy_times = []

    for _ in range(3):
        start_time = time.perf_counter()
        _ = compute_mandelbrot_naive(x_dim=x_dim, y_dim=y_dim, res=res, max_iter=max_iter)
        naive_times.append(time.perf_counter() - start_time)

        start_time = time.perf_counter()
        _ = compute_mandelbrot_numpy(x_dim=x_dim, y_dim=y_dim, res=res, max_iter=max_iter)
        numpy_times.append(time.perf_counter() - start_time)

    naive_median = statistics.median(naive_times)
    numpy_median = statistics.median(numpy_times)

    assert numpy_median < 0.1 * naive_median

#----------------------------
# Test GPU
#----------------------------

# Test 1: Integration test: the GPU float32 grid must match the serial result on a small grid
ASSEMBLED_GPU_TEST = [
    (32, (-2.0, 1.0), (-1.5, 1.5), 100)
]

@pytest.mark.parametrize('N, x_dim, y_dim, max_iter', ASSEMBLED_GPU_TEST)
def test_pixel_match_gpu32_serial(N, x_dim, y_dim, max_iter):
    result_serial = mandelbrot_serial(N=N, x_dim=x_dim, y_dim=y_dim, max_iter=max_iter)
    _, result_gpu_f32 = mandelbrot_gpu_f32(N=N, max_iter=max_iter, x_dim=x_dim, y_dim=y_dim)

    np.testing.assert_array_equal(result_serial, result_gpu_f32)

# Test 2: Integration test: the GPU float64 grid must match the serial result on a small grid
@pytest.mark.parametrize('N, x_dim, y_dim, max_iter', ASSEMBLED_GPU_TEST)
def test_pixel_match_gpu64_serial(N, x_dim, y_dim, max_iter):
    result_serial = mandelbrot_serial(N=N, x_dim=x_dim, y_dim=y_dim, max_iter=max_iter)
    _, result_gpu_f64 = mandelbrot_gpu_f64(N=N, max_iter=max_iter, x_dim=x_dim, y_dim=y_dim)

    np.testing.assert_array_equal(result_serial, result_gpu_f64)

# Test 3: Integration test: the GPU float32 grid must match the GPU float64 result on a small grid
@pytest.mark.parametrize('N, x_dim, y_dim, max_iter', ASSEMBLED_GPU_TEST)
def test_pixel_match_gpu32_gpu64(N, x_dim, y_dim, max_iter):
    _, result_gpu_f32 = mandelbrot_gpu_f32(N=N, max_iter=max_iter, x_dim=x_dim, y_dim=y_dim)
    _, result_gpu_f64 = mandelbrot_gpu_f64(N=N, max_iter=max_iter, x_dim=x_dim, y_dim=y_dim)

    np.testing.assert_array_equal(result_gpu_f32, result_gpu_f64)
