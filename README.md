# AMD gsplat

[http://www.gsplat.studio/](http://www.gsplat.studio/)

gsplat is an open-source library for CUDA accelerated rasterization of gaussians with python bindings. It is inspired by the SIGGRAPH paper [3D Gaussian Splatting for Real-Time Rendering of Radiance Fields](https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/), 

This repo is the HIP port of the gsplat repo which is enabled and optimised for RCOM runing on AMD Instinct processors. 
## System & Software Requirements
- AMD Instinct Processors - MI 300X
- Ubuntu 22.04 and above
- ROCm 6.4
- Python 3.11
  
## Installation

 - Install [Pytorch](https://pytorch.org/get-started/locally/) first. The easiet way is the use the official docker image which has pytorch for ROCm.

    ```bash
    docker run -it --network=host --device=/dev/kfd --device=/dev/dri --group-add=video --ipc=host --cap-add=SYS_PTRACE --security-opt seccomp=unconfined --shm-size 8G -v $HOME/dockerx:/dockerx -w /dockerx rocm/pytorch​:latest 
    ```
 
- Install the gsplat from the AMD hosted PYPI repoitory.

    ```bash
    pip install amd-gsplat --extra-index-url=https://pypi.amd.com/simple
    ```
## Examples

We provide a set of examples to get you started! Below you can find the details about
the examples 

- Get the examples folder from the github repo
  ```bash
  git clone --no-checkout https://github.com/rocm/gsplat.git && cd gsplat && git sparse-checkout init --cone
  git sparse-checkout add examples && git checkout main
  ```
- Install the dependencies 
   ```bash 
   cd examples
   pip install -r examples/requirements.txt
   python datasets/download_dataset.py
  ```

- [Train a 3D Gaussian splatting model on a COLMAP capture.](https://docs.gsplat.studio/main/examples/colmap.html) This example recreates a photograph using a bunch of small, colorful blobs (called "Gaussians"). Check the output in the results folder 
  ```bash
  python image_fitting.py --height 256 --width 256 --num_points 2000 --save_imgs
  ```
  
- [Fit a 2D image with 3D Gaussians.](https://docs.gsplat.studio/main/examples/image.html) This trains a 3D gaussian splatting model for novel view synthesis, on a COLMAP processed capture. A COLMAP capture is not just images; it is a complete dataset that includes original 2D images, calculated camera positions and orientations for each image, A 3D point cloud that serves as an initial guess for the scene’s geometry.
  ```bash
  CUDA_VISIBLE_DEVICES=0 python simple_trainer.py default \
    --data_dir data/360_v2/garden/ --data_factor 4 \
    --result_dir ./results/garden
  ```
- [Render a large scene in real-time.](https://docs.gsplat.studio/main/examples/large_scale.html) In this example a large scene is mimicked by replicating the Garden scene into a 9x9 grid, which results 30M Gaussians in total while gsplat still allows real-time rendering for it.
 The main magic that allows this is a very simple trick: disregard the Gaussians that are far away from the camera, by applying a small threshold (e.g., 3 pixel) to the projected Gaussian radius which is configurable in our rasterization() API as radius_clip.   
 ```bash
 # First train a 3DGS model
  CUDA_VISIBLE_DEVICES=0 python simple_trainer.py default \
    --data_dir data/360_v2/garden/ --data_factor 4 \
    --result_dir ./results/garden

  # View it in a viewer with gsplat
  python simple_viewer.py --scene_grid 5 --ckpt results/garden/ckpts/ckpt_6999.pt \
   --backend gsplat
```

## Building from source
Build steps:
- Clone the source repo from AMD
```bash
git clone https://github.com/rocm/gsplat.git
```

- Install glm and glog dependencies
   
```bash
cd gsplat/gsplat/cuda/csrc/third_party 
git clone https://github.com/google/glog.git
cd glog
cmake -S . -B build -G "Unix Makefiles" -DCMAKE_INSTALL_PREFIX:PATH=~/.local
cmake --build build --target install
git clone https://github.com/g-truc/glm
cd glm
cmake \
    -DGLM_BUILD_TESTS=OFF \
    -DBUILD_SHARED_LIBS=OFF \
    -B build . \
    -DCMAKE_INSTALL_PREFIX:PATH=~/.local
cmake --build build -- all
cmake --build build -- install
``` 
- Install Nerfacc dependency

```bash
pip install and_nerfacc --extra-index-url=https://pypi.amd.com/simple
 ```
-  Build the GSPLAT wheel
```bash
python setup.py bdist_wheel
```

-  Install the GSPLAT wheel
```bash
pip install dist/gsplat-1.0.0-cp311-cp311-linux_x86_64.whl
```
- Run a specific test or tests:
```bash
cd tests
pytest -s -v test_2dgs.py::test_fully_fused_projection_packed
pytest -s -v test_2dgs.py
pytest -s -v test_basic.py
```

## Evaluation

This repo comes with a standalone script that reproduces the official Gaussian Splatting with exactly the same performance on PSNR, SSIM, LPIPS, and converged number of Gaussians. Powered by gsplat’s efficient CUDA implementation, the training takes up to **4x less GPU memory** with up to **15% less time** to finish than the official implementation. Full report can be found [here](https://docs.gsplat.studio/main/tests/eval.html).

```bash
cd examples
pip install -r requirements.txt
# download mipnerf_360 benchmark data
python datasets/download_dataset.py
# run batch evaluation
bash benchmarks/basic.sh
```


## Contributing
We welcome contributions of any kind and are open to feedback, bug-reports, and improvements to help expand the capabilities of this software. Please check [Contributing.md](docs/Contributing.md) for more info.


## Core Development

This repository was born from the curiosity of people on the Nerfstudio team trying to understand a new rendering technique. We welcome contributions of any kind and are open to feedback, bug-reports, and improvements to help expand the capabilities of this software.

This project is developed by the following wonderful contributors (unordered):

- [Angjoo Kanazawa](https://people.eecs.berkeley.edu/~kanazawa/) (UC Berkeley): Mentor of the project.
- [Matthew Tancik](https://www.matthewtancik.com/about-me) (Luma AI): Mentor of the project.
- [Vickie Ye](https://people.eecs.berkeley.edu/~vye/) (UC Berkeley): Project lead. v0.1 lead.
- [Matias Turkulainen](https://maturk.github.io/) (Aalto University): Core developer.
- [Ruilong Li](https://www.liruilong.cn/) (UC Berkeley): Core developer. v1.0 lead.
- [Justin Kerr](https://kerrj.github.io/) (UC Berkeley): Core developer.
- [Brent Yi](https://github.com/brentyi) (UC Berkeley): Core developer.
- [Zhuoyang Pan](https://panzhy.com/) (ShanghaiTech University): Core developer.
- [Jianbo Ye](http://www.jianboye.org/) (Amazon): Core developer.

We also have a white paper with about the project with benchmarking and mathematical supplement with conventions and derivations, available [here](https://arxiv.org/abs/2409.06765). If you find this library useful in your projects or papers, please consider citing:

```
@article{ye2025gsplat,
  title={gsplat: An open-source library for Gaussian splatting},
  author={Ye, Vickie and Li, Ruilong and Kerr, Justin and Turkulainen, Matias and Yi, Brent and Pan, Zhuoyang and Seiskari, Otto and Ye, Jianbo and Hu, Jeffrey and Tancik, Matthew and Angjoo Kanazawa},
  journal={Journal of Machine Learning Research},
  volume={26},
  number={34},
  pages={1--17},
  year={2025}
}
```


## News

[May 2025] Arbitrary batching (over multiple scenes and multiple viewpoints) is supported now!! Checkout [here](docs/batch.md) for more details! Kudos to [Junchen Liu](https://junchenliu77.github.io/).

[May 2025] [Jonathan Stephens](https://x.com/jonstephens85) makes a great [tutorial video](https://www.youtube.com/watch?v=ACPTiP98Pf8) for Windows users on how to install gsplat and get start with 3DGUT.

[April 2025] [NVIDIA 3DGUT](https://research.nvidia.com/labs/toronto-ai/3DGUT/) is now integrated in gsplat! Checkout [here](docs/3dgut.md) for more details. [[NVIDIA Tech Blog]](https://developer.nvidia.com/blog/revolutionizing-neural-reconstruction-and-rendering-in-gsplat-with-3dgut/) [[NVIDIA Sweepstakes]](https://www.nvidia.com/en-us/research/3dgut-sweepstakes/)
