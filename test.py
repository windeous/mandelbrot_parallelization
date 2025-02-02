import numpy as np 
import matplotlib.pyplot as plt
import time
from multiprocessing import Pool, cpu_count
from numba import cuda

# --- Coordinate Generation Function ---

def generate_axes(real_range, imag_range, density, dtype=np.float32):

    realAxis = np.linspace(real_range[0], real_range[1], density, dtype=dtype)
    imagAxis = np.linspace(imag_range[0], imag_range[1], density, dtype=dtype)
    return realAxis, imagAxis

# --- Timing Function ---

def time_execution(func, *args, **kwargs):

    start_time = time.time()
    result = func(*args, **kwargs)
    elapsed = time.time() - start_time
    return result, elapsed

# --- Sequential Version ---

def count_iterations(c, threshold):
    z = 0j
    for iteration in range(threshold):
        z = z * z + c
        if abs(z) > 2:
            return iteration
    return threshold - 1

def mandelbrot_seq(threshold, density, real_range=(-0.22, -0.219), imag_range=(-0.70, -0.699)):
    realAxis, imagAxis = generate_axes(real_range, imag_range, density, dtype=np.float64)
    atlas = np.empty((density, density), dtype=np.int32)
    
    for ix, cx in enumerate(realAxis):
        for iy, cy in enumerate(imagAxis):
            c = complex(cx, cy)
            atlas[ix, iy] = count_iterations(c, threshold)
    return atlas

# --- Multiprocessing Version ---

def mandelbrot_row(args):
 
    row_index, realAxis, imagAxis, threshold = args
    row = np.empty(len(imagAxis), dtype=np.int32)
    cx = realAxis[row_index]
    for iy, cy in enumerate(imagAxis):
        c = complex(cx, cy)
        row[iy] = count_iterations(c, threshold)
    return row_index, row

def mandelbrot_mp(threshold, density, num_processes=None,
                  real_range=(-0.22, -0.219), imag_range=(-0.70, -0.699)):
    if num_processes is None:
        num_processes = cpu_count()
        
    realAxis, imagAxis = generate_axes(real_range, imag_range, density, dtype=np.float64)
    atlas = np.empty((density, density), dtype=np.int32)
    
    # Prepare arguments for each row
    args = [(i, realAxis, imagAxis, threshold) for i in range(density)]
    
    with Pool(processes=num_processes) as pool:
        for row_index, row in pool.imap_unordered(mandelbrot_row, args):
            atlas[row_index, :] = row

    return atlas

# --- CUDA Version ---

@cuda.jit
def mandelbrot_kernel(realAxis, imagAxis, threshold, atlas):

    ix, iy = cuda.grid(2)
    height, width = atlas.shape

    if ix < height and iy < width:
        cx = realAxis[ix]
        cy = imagAxis[iy]
        c = complex(cx, cy)
        z = 0j
        count = 0
        while count < threshold and (z.real * z.real + z.imag * z.imag) <= 4:
            z = z * z + c
            count += 1
        atlas[ix, iy] = count

def mandelbrot_cuda(threshold, density,
                    real_range=(-0.22, -0.219), imag_range=(-0.70, -0.699),
                    threadsperblock=(16, 16)):
    realAxis, imagAxis = generate_axes(real_range, imag_range, density, dtype=np.float32)
    atlas = np.empty((density, density), dtype=np.int32)
    
    d_realAxis = cuda.to_device(realAxis)
    d_imagAxis = cuda.to_device(imagAxis)
    d_atlas = cuda.to_device(atlas)
    
    blockspergrid_x = int(np.ceil(density / threadsperblock[0]))
    blockspergrid_y = int(np.ceil(density / threadsperblock[1]))
    blockspergrid = (blockspergrid_x, blockspergrid_y)
    
    mandelbrot_kernel[blockspergrid, threadsperblock](d_realAxis, d_imagAxis, threshold, d_atlas)
    
    d_atlas.copy_to_host(atlas)
    return atlas

# --- Plotting Utility ---

def plot_atlas(atlas, title="Mandelbrot Set"):
    plt.imshow(atlas.T, interpolation="nearest", origin="lower", cmap="magma")
    plt.title(title)
    plt.colorbar(label="Iteration count")
    plt.show()

# --- Benchmarking Functions ---

def benchmark_multiprocessing(threshold, density, core_list,
                              real_range=(-0.22, -0.219), imag_range=(-0.70, -0.699)):

    mp_times = []
    for n in core_list:
        # Use time_execution to measure runtime.
        _, t = time_execution(mandelbrot_mp, threshold, density, n, real_range, imag_range)
        print(f"Multiprocessing with {n} processes took {t:.4f} seconds")
        mp_times.append(t)
    return mp_times

def benchmark_cuda_parallel(threshold, density, thread_configs, 
                            real_range=(-0.22, -0.219), imag_range=(-0.70, -0.699)):

    cuda_results = []
    
    for config in thread_configs:
        try:
            _, t = time_execution(mandelbrot_cuda, threshold, density, real_range, imag_range, config)
            total_threads = config[0] * config[1]
            print(f"CUDA with threads per block {config} (total threads {total_threads}) took {t:.4f} seconds")
            cuda_results.append((total_threads, t))
        except cuda.CudaSupportError as e:
            print("CUDA is not available on this system:", e)
            return None
    return cuda_results

def run_benchmarks(threshold, density):

    # --- Sequential benchmark ---
    print("Running sequential version for benchmarking...")
    _, sequential_time = time_execution(mandelbrot_seq, threshold, density)
    print(f"Sequential version took {sequential_time:.4f} seconds")
    
    # --- Multiprocessing benchmark ---
    max_cores = cpu_count()
    core_list = list(range(2, max_cores+1))
    print("\nStarting multiprocessing benchmark...")
    mp_times = benchmark_multiprocessing(threshold, density, core_list)
    
    # --- CUDA benchmark ---
    cuda_thread_configs = [
        (1, 1),
        (2, 2),
        (4, 4), 
        (16, 16), 
        (32, 32)
        ]
    print("\nStarting CUDA benchmark...")
    cuda_results = benchmark_cuda_parallel(threshold, 10000, cuda_thread_configs)
    
    plt.figure(figsize=(10, 6))
    
    plt.hlines(sequential_time, core_list[0], core_list[-1], colors='g', 
               label=f'Sequential ({sequential_time:.4f}s)')
    
    plt.plot(core_list, mp_times, marker='o', label='Multiprocessing')
    for x_pos, time in zip(core_list, mp_times):
            plt.annotate(f"{time: .2f}s", (x_pos, time), textcoords="offset points", xytext=(0,8), ha="center")
    
    if cuda_results:
        cuda_results.sort(key=lambda x: x[0])
        cuda_threads, cuda_times = zip(*cuda_results)
        cuda_x_positions = np.linspace(min(core_list), max(core_list), len(cuda_results))

        plt.plot(cuda_x_positions, cuda_times, marker='s', label='CUDA') 

        for x_pos, (threads, time) in zip(cuda_x_positions, cuda_results):
            plt.annotate(f"{time: .2f}s\n{threads} threads", (x_pos, time), textcoords="offset points", xytext=(20,-15), ha="center")
    
    plt.xlabel("Number of Processes (or Total Threads for CUDA)")
    plt.ylabel("Execution Time (seconds)")
    plt.title("Mandelbrot Benchmark: Sequential vs Multiprocessing vs CUDA")
    plt.legend()
    plt.grid(True)
    plt.show()


if __name__ == "__main__":
    # Adjust these parameters as needed. Threshold is how many times a pixel is reiterated before being placed
    threshold = 500 #how many times each point is iterated over before becoming part of the mandelbrot set
    density = 1000 #how many points there are (pixels in the fractal image)

    print("Running sequential version...")
    atlas_seq, time_seq = time_execution(mandelbrot_seq, threshold, density)
    print(f"Sequential version took {time_seq:.4f} seconds")
    plot_atlas(atlas_seq, title="Mandelbrot Set (Sequential)")

    print("Running multiprocessing version (4 cores)...")
    atlas_mp, time_mp = time_execution(mandelbrot_mp, threshold, density, 4)
    print(f"Multiprocessing version took {time_mp:.4f} seconds")
    plot_atlas(atlas_mp, title="Mandelbrot Set (Multiprocessing)")

    print("Running CUDA version...")
    try:
        atlas_cuda, time_cuda = time_execution(mandelbrot_cuda, threshold, density=10000)
        print(f"CUDA version took {time_cuda:.4f} seconds")
        plot_atlas(atlas_cuda, title="Mandelbrot Set (CUDA)")
    except cuda.CudaSupportError as e:
        print("CUDA is not available on this system:", e)

    # --- run the benchmarks ---
    print("\nRunning combined benchmarks for sequential, multiprocessing, and CUDA versions...")
    run_benchmarks(threshold, density)
