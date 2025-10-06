# gsplat for ROCm

**gsplat** is an open-source library for GPU-accelerated rasterization of Gaussians with Python bindings. It is inspired by the SIGGRAPH paper [3D Gaussian Splatting for Real-Time Rendering of Radiance Fields](https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/).

This repository is the HIP port of the original `gsplat` project, optimized for **ROCm**, and designed to run on AMD Instinct™ GPUs. 

## System Requirements

To use gsplat, you need the following prerequisites:

- **ROCm**: version 6.4.3 (recommended)
- **Operating system**: Ubuntu 24.04  
- **GPU platform**: AMD Instinct™ MI300X  
- **PyTorch**: version 2.6 (ROCm-enabled)  
- **Python**: version 3.12  

## Installation

1. Install PyTorch (with ROCm support).  
   The easiest method is using the official ROCm PyTorch Docker image:

   ```bash
   docker pull rocm/pytorch:rocm6.4.3_ubuntu24.04_py3.12_pytorch_release_2.6.0
   ```

2. Launch and connect to the container:

   ```bash
   docker run --cap-add=SYS_PTRACE --ipc=host --privileged=true      --shm-size=128GB --network=host      --device=/dev/kfd --device=/dev/dri      --group-add video -it -v $HOME:$HOME      --name rocm_pytorch rocm/pytorch:rocm6.4.3_ubuntu24.04_py3.12_pytorch_release_2.6.0
   ```

3. Install gsplat from the AMD-hosted PyPI repository:

   ```bash
   pip install gsplat --index-url=https://pypi.amd.com/simple
   ```

4. Verify the installation:

   ```bash
   pip show gsplat
   ```

5. The output should show as follows:

   ```bash
   Name: gsplat
   Version: 1.0.0+4ae1c82
   Summary: Python package for differentiable rasterization of Gaussians
   Home-page: https://github.com/rocm/gsplat
   Author: AMD Corporation
   License: Apache 2.0
   Location: /opt/conda/envs/py_3.12/lib/python3.12/site-packages
   Requires: jaxtyping, ninja, numpy, rich, torch


## Examples

We provide a set of examples to get you started. 

1. Clone the examples folder:

   ```bash
   git clone --no-checkout https://github.com/rocm/gsplat.git
   cd gsplat
   git sparse-checkout init --cone
   git sparse-checkout add examples
   git checkout main
   ```

2. Install dependencies and download datasets:

   ```bash
   cd examples
   pip install -r requirements.txt
   python datasets/download_dataset.py
   ```

3. To run the examples, refer to the [run a gsplat example](docs/examples/gsplat-examples.rst) topic. The examples are as follows:

- [Fit a Single Image](docs/examples/gsplat-examples.rst#fit-a-single-image)
- [Fit a 2D image with 3D Gaussians](docs/examples/gsplat-examples.rst#fit-a-single-2d-image-with-3d-gaussians)
- [Render a large scene in real-time](docs/examples/gsplat-examples.rst#render-a-large-scene-in-real-time)

## Evaluation

This repository includes a standalone script that reproduces the official Gaussian Splatting benchmarks with equivalent performance on **PSNR, SSIM, LPIPS**, and the number of converged Gaussians.  

Thanks to gsplat’s optimized GPU implementation:  
- Training uses up to **4× less GPU memory**  
- Training is up to **15% faster** compared to the official implementation  

## Building from source
Refer to the [installation instructions](docs/install/gsplat-install.rst) to learn how to build the gsplat library from source.
