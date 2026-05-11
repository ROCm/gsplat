"""
Patch GLM's platform.h to check HIP before CUDA.

PyTorch's ROCm hipification converts __CUDACC__ to __HIP__ in GLM headers.
Since GLM checks CUDA before HIP in the original code, the hipified version's
converted CUDA block shadows the native HIP detection, causing a build failure
("GLM requires CUDA 7.0 or higher") because CUDA_VERSION is not defined on HIP.

This patch swaps the order so the native __HIP__ check comes first, ensuring
that even after hipification the HIP compiler path is correctly selected.
"""

import os.path as osp
import re

GLM_PLATFORM_H = osp.join(
    osp.dirname(__file__),
    "..",
    "gsplat",
    "cuda",
    "csrc",
    "third_party",
    "glm",
    "glm",
    "simd",
    "platform.h",
)


def patch_platform_h(path: str) -> bool:
    with open(path, "r") as f:
        content = f.read()

    # Match the CUDA block followed by the HIP block
    # We swap them so HIP is checked first.
    cu_block = r"""(// CUDA
#elif defined\(__CUDACC__\))
#\tif !defined\(CUDA_VERSION\) && !defined\(GLM_FORCE_CUDA\)
#\t\tinclude <cuda\.h>  // make sure version is defined since nvcc does not define it itself!
#\tendif
#\tif defined\(__CUDACC_RTC__\)
#\t\tdefine GLM_COMPILER GLM_COMPILER_CUDA_RTC
#\telif CUDA_VERSION >= 8000
#\t\tdefine GLM_COMPILER GLM_COMPILER_CUDA80
#\telif CUDA_VERSION >= 7500
#\t\tdefine GLM_COMPILER GLM_COMPILER_CUDA75
#\telif CUDA_VERSION >= 7000
#\t\tdefine GLM_COMPILER GLM_COMPILER_CUDA70
#\telif CUDA_VERSION < 7000
#\t\terror "GLM requires CUDA 7\.0 or higher"
#\tendif

// HIP
#elif defined\(__HIP__\)
#\tdefine GLM_COMPILER GLM_COMPILER_HIP"""

    hip_block = """// HIP
#elif defined(__HIP__)
#\tdefine GLM_COMPILER GLM_COMPILER_HIP"""

    cu_block_only = """// CUDA
#elif defined(__CUDACC__)
#\tif !defined(CUDA_VERSION) && !defined(GLM_FORCE_CUDA)
#\t\tinclude <cuda.h>  // make sure version is defined since nvcc does not define it itself!
#\tendif
#\tif defined(__CUDACC_RTC__)
#\t\tdefine GLM_COMPILER GLM_COMPILER_CUDA_RTC
#\telif CUDA_VERSION >= 8000
#\t\tdefine GLM_COMPILER GLM_COMPILER_CUDA80
#\telif CUDA_VERSION >= 7500
#\t\tdefine GLM_COMPILER GLM_COMPILER_CUDA75
#\telif CUDA_VERSION >= 7000
#\t\tdefine GLM_COMPILER GLM_COMPILER_CUDA70
#\telif CUDA_VERSION < 7000
#\t\terror "GLM requires CUDA 7.0 or higher"
#\tendif"""

    # Build the replacement: HIP block first, then CUDA
    new_hip_block = hip_block.replace("#\t", "\t")
    new_cu_block = cu_block_only.replace("#\t", "\t")

    replacement = f"{new_hip_block}\n\n{new_cu_block}"

    # Match the full original order: CUDA then HIP
    # We use a simpler approach: find the exact text and replace
    old = f"""// CUDA
#elif defined(__CUDACC__)
#\tif !defined(CUDA_VERSION) && !defined(GLM_FORCE_CUDA)
#\t\tinclude <cuda.h>  // make sure version is defined since nvcc does not define it itself!
#\tendif
#\tif defined(__CUDACC_RTC__)
#\t\tdefine GLM_COMPILER GLM_COMPILER_CUDA_RTC
#\telif CUDA_VERSION >= 8000
#\t\tdefine GLM_COMPILER GLM_COMPILER_CUDA80
#\telif CUDA_VERSION >= 7500
#\t\tdefine GLM_COMPILER GLM_COMPILER_CUDA75
#\telif CUDA_VERSION >= 7000
#\t\tdefine GLM_COMPILER GLM_COMPILER_CUDA70
#\telif CUDA_VERSION < 7000
#\t\terror "GLM requires CUDA 7.0 or higher"
#\tendif

// HIP
#elif defined(__HIP__)
#\tdefine GLM_COMPILER GLM_COMPILER_HIP"""

    new = f"""// HIP
#elif defined(__HIP__)
#\tdefine GLM_COMPILER GLM_COMPILER_HIP

// CUDA
#elif defined(__CUDACC__)
#\tif !defined(CUDA_VERSION) && !defined(GLM_FORCE_CUDA)
#\t\tinclude <cuda.h>  // make sure version is defined since nvcc does not define it itself!
#\tendif
#\tif defined(__CUDACC_RTC__)
#\t\tdefine GLM_COMPILER GLM_COMPILER_CUDA_RTC
#\telif CUDA_VERSION >= 8000
#\t\tdefine GLM_COMPILER GLM_COMPILER_CUDA80
#\telif CUDA_VERSION >= 7500
#\t\tdefine GLM_COMPILER GLM_COMPILER_CUDA75
#\telif CUDA_VERSION >= 7000
#\t\tdefine GLM_COMPILER GLM_COMPILER_CUDA70
#\telif CUDA_VERSION < 7000
#\t\terror "GLM requires CUDA 7.0 or higher"
#\tendif"""

    if old in content:
        content = content.replace(old, new)
        with open(path, "w") as f:
            f.write(content)
        print(f"Patched {path} — swapped HIP check before CUDA check")
        return True
    else:
        print(f"Warning: Could not find the CUDA/HIP block pattern in {path}")
        print("The file may already be patched or have a different version.")
        return False


if __name__ == "__main__":
    import sys

    path = sys.argv[1] if len(sys.argv) > 1 else GLM_PLATFORM_H
    success = patch_platform_h(path)
    sys.exit(0 if success else 1)
