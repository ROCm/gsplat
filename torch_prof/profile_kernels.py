#!/usr/bin/env python3
"""Profile individual gsplat CUDA/HIP kernels."""

import torch
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from torch_prof.profiler import profile_code
from gsplat._helper import load_test_data
from gsplat.rendering import rasterization
from gsplat.cuda._wrapper import (
    quat_scale_to_covar_preci,
    proj,
    fully_fused_projection,
    isect_tiles,
    isect_offset_encode,
    rasterize_to_pixels,
)
import numpy as np


def profile_individual_kernels():
    """Profile individual CUDA/HIP kernels used in gsplat."""
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load test data
    (
        means,
        quats,
        scales,
        opacities,
        colors,
        viewmats,
        Ks,
        width,
        height,
    ) = load_test_data(
        data_path=os.path.join(os.path.dirname(__file__), "../assets/test_garden.npz"),
        device=device
    )
    
    # Prepare common inputs
    means = means.contiguous().requires_grad_(True)
    quats = quats.contiguous().requires_grad_(True)
    scales = scales.contiguous().requires_grad_(True)
    opacities = opacities.contiguous().requires_grad_(True)
    colors = colors.contiguous().requires_grad_(True)
    viewmat = viewmats[0:1]
    K = Ks[0:1]
    
    print("Profiling individual gsplat kernels...")
    
    # 1. Profile quat_scale_to_covar_preci
    with profile_code("quat_scale_to_covar_preci", group="kernels"):
        for _ in range(10):
            covars, preci = quat_scale_to_covar_preci(quats, scales, triu=False)
            torch.cuda.synchronize()
    
    # 2. Profile projection
    with profile_code("projection", group="kernels"):
        for _ in range(10):
            proj_results = proj(means, None, K, viewmat, width, height)
            torch.cuda.synchronize()
    
    # 3. Profile fully_fused_projection
    with profile_code("fully_fused_projection", group="kernels"):
        for _ in range(10):
            results = fully_fused_projection(
                means, None, quats, scales, viewmat, K,
                width, height
            )
            torch.cuda.synchronize()
    
    # 4. Profile tile intersection
    radii = torch.ones(len(means), dtype=torch.int32, device=device) * 10
    camera_ids = torch.zeros(len(means), dtype=torch.int64, device=device)
    gaussian_ids = torch.arange(len(means), dtype=torch.int32, device=device)
    
    with profile_code("isect_tiles", group="kernels"):
        for _ in range(10):
            tiles_per_gauss, isect_ids, flatten_ids = isect_tiles(
                means2d=means[:, :2],
                radii=radii,
                depths=means[:, 2],
                tile_size=16,
                tile_width=width // 16,
                tile_height=height // 16,
                packed=False,
                n_cameras=1,
                camera_ids=camera_ids,
                gaussian_ids=gaussian_ids,
            )
            torch.cuda.synchronize()
    
    # 5. Profile isect_offset_encode
    with profile_code("isect_offset_encode", group="kernels"):
        for _ in range(10):
            isect_offsets = isect_offset_encode(isect_ids, 1, width // 16, height // 16)
            torch.cuda.synchronize()
    
    # 6. Profile rasterization
    with profile_code("rasterize_to_pixels", group="kernels"):
        # Prepare inputs for rasterization
        transmittances = torch.ones(height * width, device=device)
        
        for _ in range(10):
            renders, alphas = rasterize_to_pixels(
                means2d=means[:, :2],
                conics=torch.rand(len(means), 3, device=device),
                colors=colors,
                opacities=opacities,
                image_width=width,
                image_height=height,
                tile_size=16,
                isect_offsets=isect_offsets,
                flatten_ids=flatten_ids,
                backgrounds=torch.zeros(1, 3, device=device),
                packed=False,
                absgrad=False,
            )
            torch.cuda.synchronize()
    
    print("\nKernel profiling complete!")


def profile_end_to_end_rasterization():
    """Profile the complete rasterization pipeline."""
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load test data
    (
        means,
        quats,
        scales,
        opacities,
        colors,
        viewmats,
        Ks,
        width,
        height,
    ) = load_test_data(
        data_path=os.path.join(os.path.dirname(__file__), "../assets/test_garden.npz"),
        device=device
    )
    
    print("\nProfiling end-to-end rasterization...")
    
    # Profile different configurations
    configs = [
        {"packed": True, "sparse_grad": True, "name": "packed_sparse"},
        {"packed": True, "sparse_grad": False, "name": "packed_dense"},
        {"packed": False, "sparse_grad": False, "name": "unpacked_dense"},
    ]
    
    for config in configs:
        with profile_code(f"rasterization_{config['name']}", group="end_to_end"):
            for _ in range(5):
                renders, meta = rasterization(
                    means=means,
                    quats=quats,
                    scales=scales,
                    opacities=opacities,
                    colors=colors,
                    viewmats=viewmats[0:1],
                    Ks=Ks[0:1],
                    width=width,
                    height=height,
                    packed=config["packed"],
                    sparse_grad=config["sparse_grad"],
                )
                
                # Backward pass
                loss = renders.sum()
                loss.backward()
                torch.cuda.synchronize()


def main():
    print("Starting kernel profiling...")
    
    # Create output directory
    os.makedirs("torch_prof/logs/kernels", exist_ok=True)
    
    # Profile individual kernels
    profile_individual_kernels()
    
    # Profile end-to-end
    profile_end_to_end_rasterization()
    
    print("\nAll profiling complete!")
    print("View results with: tensorboard --logdir=torch_prof/logs/kernels")


if __name__ == "__main__":
    main()