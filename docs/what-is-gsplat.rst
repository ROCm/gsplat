.. meta::
  :description: What is gsplat?
  :keywords: gaussian splatting, gsplat, AMD, ROCm, overview, introduction

.. _what-is-gsplat:

********************************************************************
What is gsplat?
********************************************************************

`gsplat <https://docs.gsplat.studio/main/>`__ (Gaussian splatting) is an open-source library for GPU-accelerated, differentiable rasterization of 3D Gaussians, with Python bindings. 
It is based on the SIGGRAPH paper `3D Gaussian Splatting for Real-Time Rendering of Radiance Fields <https://doi.org/10.1145/3588432.3591490>`__.
gsplat provides an optimized implementation with improved performance, reduced memory usage, and faster convergence. 
The library features a modular API with Python support, suitable for both research and practical applications.

Gaussian splatting overview
====================================================================

Gaussian splatting is an efficient technique for real-time rendering of 3D scenes.  
It has emerged as an alternative to neural radiance fields (NeRFs), offering advantages in rendering speed and quality.  

Unlike NeRFs, which represent a scene as a neural network, Gaussian splatting models a scene as a collection of 3D Gaussians; ellipsoidal shapes with associated color and opacity properties.  
This representation allows for rapid rendering, making it suitable for interactive applications such as virtual reality (VR), augmented reality (AR), and video games.

Features and use cases
====================================================================

**Real-time rendering**: Gaussian splatting renders high-quality 3D scenes interactively by representing the scene with a discrete set of 3D Gaussians.  
The rendering process becomes a parallelizable and efficient splatting operation, well-suited for GPU architectures.  
This speed supports applications where frame rate is critical.

**High fidelity and quality**: Gaussian splatting can produce detailed scenes.  
The 3D Gaussians capture geometric and photometric details, including textures, reflections, and lighting effects, resulting in realistic outputs.

**Explicit scene representation**: Unlike NeRFs, Gaussian splatting uses an explicit, discrete set of primitives.  
This makes the representation interpretable and editable.  
You can add, remove, or modify Gaussians, which provides flexibility compared to implicit neural representations.

**Efficient training and optimization**: Creating a gsplat model from images is efficient.  
The optimization process adjusts Gaussian parameters (position, shape, color, opacity) to best match the input.  
Training often takes minutes, compared with hours for NeRF models.

**Compact scene representation**: A gsplat model consists only of Gaussian parameters, making it relatively small in size.  
This compact format simplifies storage, sharing, and deployment, which is especially important for mobile or web delivery.

Performance
====================================================================

gsplat is developed with efficiency in mind:  

- On AMD Instinct™ GPUs, you can leverage greater GPU memory to train larger scenes. See :doc:`benchmarking <reference/benchmark-evaluation>` for more details.
- gsplat supports large scene rendering and performs faster than the official `https://github.com/graphdeco-inria/diff-gaussian-rasterization <https://github.com/graphdeco-inria/diff-gaussian-rasterization>`__.  
- Extra features include ``batch rasterization``, ``N-D feature rendering``, ``depth rendering``, ``sparse gradients``, and ``multi-GPU distributed rasterization``.  
- gsplat integrates techniques such as ``absgrad``, ``anti-aliasing``, and ``3DGS-MCMC``. See the :doc:`API reference <reference/gsplat-api-reference>` for more details. 
