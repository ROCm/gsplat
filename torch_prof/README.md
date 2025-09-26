# GSplat PyTorch Profiling Framework

A comprehensive profiling framework for gsplat using PyTorch Profiler, designed for AMD MI300X GPUs with ROCm support.

## Overview

This framework provides automated profiling for:
- All test files (`test_*.py`)
- Benchmark script (`profiling/main.py`)

All profiling traces are saved to `torch_prof/logs/` and can be viewed with TensorBoard or Chrome tracing.

## Quick Usage


### Profile Tests Only
```bash
# Profile all test files
python torch_prof/profile_tests.py --tests test_basic.py test_rasterization.py

# Profile specific test
python torch_prof/profile_tests.py --tests test_basic.py --specific-test test_rasterize_to_pixels
```

### Profile Benchmark Script Only
```bash
# Profile main.py benchmark
python torch_prof/profile_tests.py --include-main
```

### Profile Everything
```bash
# Profile all components
python torch_prof/profile_tests.py --include-main --tests test_basic.py
```

## Detailed Usage

### Profile All Tests

```bash
# Run profiling on all tests (excluding compression)
CUDA_VISIBLE_DEVICES=0 python torch_prof/profile_tests.py --tests test_basic.py test_rasterization.py test_2dgs.py test_strategy.py

# Profile specific test with pattern matching
CUDA_VISIBLE_DEVICES=0 python torch_prof/profile_tests.py --tests test_basic.py --specific-test "test_rasterize_to_pixels[batch_dims2-128]"
```

### Profile Main Benchmark Script

```bash
# Without profiling (normal benchmark)
CUDA_VISIBLE_DEVICES=0 python profiling/main.py --batch_size 8 --scene_grid 21 --channels 32 --resolution 720p

# With profiling enabled
ENABLE_PROFILER=1 CUDA_VISIBLE_DEVICES=0 python profiling/main.py --batch_size 8 --scene_grid 21 --channels 32 --repeats 5 --resolution 4k
```

### Profile Individual Kernels

```bash
# Profile individual CUDA/HIP kernels and end-to-end pipeline
CUDA_VISIBLE_DEVICES=0 python torch_prof/profile_kernels.py
```

## Analyzing Results

### 1. Extract Key Performance Metrics
```bash
# Get total_direct_kernel_time_ms for all traces
python torch_prof/extract_kernel_times.py

# Extract from specific trace
python torch_prof/extract_kernel_times.py torch_prof/logs/tests/test_strategy/trace.pt.trace.json
```

### 2. Generate Excel Reports with TraceLens
```bash
# Generate TraceLens Excel reports for all traces
python torch_prof/use_tracelens.py --all

# Generate report for specific trace
python torch_prof/use_tracelens.py --trace path/to/trace.pt.trace.json

# List available traces
python torch_prof/use_tracelens.py --list
```

### 3. View in Browser (Recommended)
```bash
# Option 1: Chrome Tracing (Most Reliable)
# 1. Open Chrome browser
# 2. Go to: chrome://tracing
# 3. Load any .pt.trace.json file

# Option 2: Perfetto (Online)
# 1. Go to: https://ui.perfetto.dev/
# 2. Drag and drop trace file
```

### 4. TensorBoard (If Available)
```bash
# Fix NumPy compatibility first
pip install "numpy<2.0"

# View all results
tensorboard --logdir=torch_prof/logs --bind_all

# View specific category
tensorboard --logdir=torch_prof/logs/tests
tensorboard --logdir=torch_prof/logs/mainbench
tensorboard --logdir=torch_prof/logs/kernels
```

## Maximum Parameter Configurations

### Test Parameters
- **Batch Dimensions**: `batch_dims2` (most complex tensor shapes)
- **Tile Size**: `128` (largest tile size)
- **SH Degree**: `3` (maximum spherical harmonics)
- **Render Mode**: `RGB+D` (most complex rendering)
- **Channels**: `31` or `32` (maximum color channels)

### Main Benchmark Parameters
- **Batch Size**: `8` (multiple views simultaneously)
- **Scene Grid**: `21` (largest scene multiplier, ~49M Gaussians)
- **Channels**: `32` (maximum color channels)
- **Repeats**: `5` (multiple runs for consistent timing)

## Environment Variables

- **`ENABLE_PROFILER=1`** - Enable torch profiler in tests and main.py
- **`CUDA_VISIBLE_DEVICES=N`** - Select GPU device (works with ROCm)
- **`HIP_VISIBLE_DEVICES=N`** - Alternative for AMD GPUs

## Output Structure

```
torch_prof/
├── logs/
│   ├── tests/           # Test profiling results
│   │   ├── test_rasterize_to_pixels_batch_dims2_128_/
│   │   └── test_strategy/
│   ├── kernels/         # Individual kernel profiling
│   │   ├── quat_scale_to_covar_preci/
│   │   └── rasterization_packed_sparse/
│   ├── mainbench/       # Main benchmark profiling
│   │   ├── mainbench_batch8_grid21_ch32_packed_True_sparse_True/
│   │   └── mainbench_batch8_grid21_ch32_packed_False_sparse_False/
│   └── end_to_end/      # End-to-end pipeline profiling
├── tracelens_reports/   # Generated Excel reports
└── kernel_times_summary.csv  # Performance summary
```

## Key Files

### Core Scripts (Keep)
- **`profiler.py`** - Core profiling utilities
- **`profile_tests.py`** - Test automation 
- **`profile_kernels.py`** - Kernel profiling
- **`use_tracelens.py`** - TraceLens integration
- **`extract_kernel_times.py`** - Performance metrics extraction
- **`run_max_params.py`** - Maximum parameter automation

### Generated Results (Can Delete/Regenerate)
- **`logs/`** - Profiling traces
- **`tracelens_reports/`** - Excel reports
- **`kernel_times_summary.csv`** - Performance summary

## Notes for AMD GPUs

- Use `CUDA_VISIBLE_DEVICES` (PyTorch ROCm maintains compatibility)
- All profiling works on AMD MI300X with ROCm
- Chrome tracing and TraceLens work best for visualization
- TensorBoard may have compatibility issues with NumPy 2.0