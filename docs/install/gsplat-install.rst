.. meta::
  :description: installing gsplat for ROCm
  :keywords: installation instructions, gsplat, AMD, ROCm, Gaussian splatting

.. _gsplat-on-rocm-installation:

********************************************************************
gsplat on ROCm installation
********************************************************************

System requirements
====================================================================

To use gsplat (Gaussian splatting), you need the following prerequisites:

- ROCm: version 6.4.3 (recommended)
- Operating system: Ubuntu 24.04  
- GPU platform: AMD Instinct™ MI300X  
- PyTorch: version 2.6 (ROCm-enabled)  
- Python: version 3.12  

Install gsplat
====================================================================

To install gsplat for ROCm, you have the following options:

* :ref:`using-docker-image-with-pytorch` **(recommended)**
* :ref:`build-from-source`

.. _using-docker-image-with-pytorch:

Using Docker with requirements preinstalled
--------------------------------------------------------------------

You need to install PyTorch to begin. The easiest way is to use the official ROCm-enabled PyTorch Docker image. 
Docker is the recommended method to set up your environment, as it avoids potential installation issues.

1. Download the latest compatible public `PyTorch Docker image <https://hub.docker.com/layers/rocm/pytorch/rocm6.4.3_ubuntu22.04_py3.10_pytorch_release_2.6.0/>`__.

   .. code-block:: bash

      docker pull rocm/pytorch:rocm6.4.3_ubuntu24.04_py3.12_pytorch_release_2.6.0

2. Launch and connect to the container

   .. code-block:: bash

       docker run --cap-add=SYS_PTRACE --ipc=host --privileged=true \
        --shm-size=128GB --network=host --device=/dev/kfd --device=/dev/dri \
        --group-add video -it -v $HOME:$HOME --name rocm_pytorch 
        rocm/pytorch:rocm6.4.3_ubuntu24.04_py3.12_pytorch_release_2.6.0

3. After setting up the container, install gsplat from the AMD-hosted `PyPI repository <https://pypi.amd.com/simple/>`__:

   .. code-block:: bash

      pip install gsplat --index-url=https://pypi.amd.com/simple

4. Verify the installation:

   .. code-block:: bash

      pip show gsplat

5. The output should show as follows:

   .. code-block:: text

      Name: gsplat
      Version: 1.0.0+4ae1c82
      Summary: Python package for differentiable rasterization of Gaussians
      Home-page: https://github.com/rocm/gsplat
      Author: AMD Corporation
      License: Apache 2.0
      Location: /opt/conda/envs/py_3.12/lib/python3.12/site-packages
      Requires: jaxtyping, ninja, numpy, rich, torch

.. _build-from-source:

Build gsplat from source
--------------------------------------------------------------------

You can choose to setup the development environment and build the gsplat library manually with the following steps.

Prerequisites
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Before building gsplat from source, ensure you have the following installed:

- `ROCm 6.4.3 <https://rocm.docs.amd.com/projects/install-on-linux/en/docs-6.4.3/>`__  
- `PyTorch with ROCm 6.4.3 <https://hub.docker.com/layers/rocm/pytorch/rocm6.4.3_ubuntu22.04_py3.10_pytorch_release_2.6.0/>`__ 

1. To verify your setup, check the ROCm installation and GPU architecture:

   .. code-block:: bash

      rocminfo | grep gfx

2. The output should be as follows: 

   .. code-block:: bash

      Name:                    gfx942
            Name:                    amdgcn-amd-amdhsa--gfx942:sramecc+:xnack-
            Name:                    amdgcn-amd-amdhsa--gfx9-4-generic:sramecc+:xnack-

3. Confirm that PyTorch is installed and accessible:

   .. code-block:: bash

      python3 -c 'import torch' 2> /dev/null && echo 'Success' || echo 'Failure'

4. Confirm that the GPU is accessible from PyTorch:

   .. code-block:: bash

      python3 -c 'import torch; print(torch.cuda.is_available())'

Build steps
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

1. Clone the repository with submodules:

   .. code-block:: bash

      git clone --recurse-submodules https://github.com/rocm/gsplat.git

2. Install the GLM dependency:

   .. code-block:: bash

      cd gsplat/cuda/csrc/third_party/glm
      cmake \
          -DGLM_BUILD_TESTS=OFF \
          -DBUILD_SHARED_LIBS=OFF \
          -B build . \
          -DCMAKE_INSTALL_PREFIX:PATH=~/.local
      cmake --build build -- all
      cmake --build build -- install

3. Build the gsplat wheel:

   .. code-block:: bash

      cd ../../../..
      python setup.py bdist_wheel

4. Install the wheel:

   .. code-block:: bash

      pip install dist/gsplat*.whl

5. Verify the installation:

   .. code-block:: bash

      pip show gsplat

Run unit tests and verify the installation
====================================================================

The ``tests/`` folder provides automatic test scripts that verify if the GPU implementations agree with those of PyTorch. 
They primarily focus on validating the functionality of the Gaussian Splatting (GS) and 3D rendering components. 
These tests ensure the correctness, performance, and stability of the core features implemented in the gsplat library.

1. To run the tests, first install the `nerfacc <https://github.com/nerfstudio-project/nerfacc>`__ dependency:

   .. code-block:: bash

      git clone https://github.com/nerfstudio-project/nerfacc.git
      cd nerfacc
      python setup.py bdist_wheel
      pip install dist/nerfacc*.whl

2. Run specific tests from the ``tests`` directory:

   .. code-block:: bash

      cd tests
      pytest -s -v test_2dgs.py::test_fully_fused_projection_packed_2dgs
      pytest -s -v test_2dgs.py
      pytest -s -v test_basic.py

Run a gsplat example
====================================================================

A set of examples is available to help you get started. See :doc:`run a gsplat example <../examples/gsplat-examples>` for more details.

Benchmarking and evaluation
====================================================================

The `https://github.com/ROCm/gsplat <https://github.com/ROCm/gsplat>`_ repository includes a standalone script that reproduces the 
official Gaussian Splatting results with matching performance on ``PSNR``, ``SSIM``, ``LPIPS``, and the converged number of Gaussians.
See :doc:`benchmarks <../reference/benchmark-evaluation>` for more details.

Known issues and limitations
-------------------------------------------------------------------

- Compression of ``splat parameters`` (``positions``, ``scales``, ``rotations``, ``colors``, and ``features``) using PNG image encoding is not supported.  
- Usage of ``Fused SSIM`` and ``Fused Bilagrid`` in the simple trainer is not supported (torch-based implementations are used instead).  
- Distributed training with the ``packed`` option is not supported. 
