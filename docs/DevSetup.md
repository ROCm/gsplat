# Building from source

## Prerequisites 

Before running the steps, ensure that you have [ROCm](https://rocm.docs.amd.com/projects/install-on-linux/en/latest/install/quick-start.html) and [Pytorch](https://rocm.docs.amd.com/projects/install-on-linux/en/latest/install/3rd-party/pytorch-install.html) installed on your system. You can verify these by running the commands as below.
  - rocminfo shoud print the gfx architecture

    ```bash
      rocminfo | grep gfx
  - Import the torch package in Python to test if PyTorch is installed and accessible.
    ```bash
    python3 -c 'import torch' 2> /dev/null && echo 'Success' || echo 'Failure'
    ```
  - Check if the GPU is accessible from PyTorch. In the PyTorch framework, torch.cuda is a generic way to access the GPU. This can only access an AMD GPU if one is available.
    ```bash
    python3 -c 'import torch; print(torch.cuda.is_available())'
    ```

## Build steps:
- Clone the source repo from AMD
    ```bash
      git clone --recurse-submodules https://github.com/rocm/gsplat.git
    ```

- Install glm dependency
   
  ```bash
  apt-get update && apt-get install -y cmake
  cd gsplat/cuda/csrc/third_party/glm
  cmake \
      -DGLM_BUILD_TESTS=OFF \
      -DBUILD_SHARED_LIBS=OFF \
      -B build . \
      -DCMAKE_INSTALL_PREFIX:PATH=~/.local
  cmake --build build -- all
  cmake --build build -- install
  ``` 
-  Build the GSPLAT wheel
    ```bash
    python setup.py bdist_wheel
    ```

-  Install the GSPLAT wheel
    ```bash
    pip install dist/gsplat*.whl
    ```
- Verify the installation using the pip show command
    ```bash
    pip show gsplat
  ```
## Running the unit tests

- Install Nerfacc dependency to run the tests

  ```bash
  git clone https://github.com/AMD-AIOSS/nerfacc.git
  python setup.py bdist_wheel
  pip install dist/nerfacc*.whl
  ```

- Run a specific test or tests:
  ```bash
  cd tests
  pytest -s -v test_2dgs.py::test_fully_fused_projection_packed_2dgs
  pytest -s -v test_2dgs.py
  pytest -s -v test_basic.py
  ```
