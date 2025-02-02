# Mandelbrot Set Parallelization

## Overview
This project implements and benchmarks different parallel computing techniques for generating the Mandelbrot set. It compares three approaches:

1. **Sequential Execution:** A single-threaded implementation for baseline performance comparison.
2. **Multiprocessing:** Uses Python’s `multiprocessing` module to distribute computations across multiple CPU cores.
3. **CUDA Acceleration:** Leverages NVIDIA GPUs via CUDA for high-performance fractal generation.

## Features
- Sequential, CPU-parallel, and GPU-parallel Mandelbrot set computation.
- Benchmarking utilities to measure performance improvements.
- Visualization tools for rendering fractal images.
- Configurable parameters for grid density, iteration thresholds, and thread distribution.

## Installation
### Prerequisites
Ensure you have the following dependencies installed:
- Python 3.x
- NumPy
- Matplotlib
- Numba (for CUDA support)
- Multiprocessing module (included with Python)
- NVIDIA GPU with CUDA support (for GPU acceleration)
