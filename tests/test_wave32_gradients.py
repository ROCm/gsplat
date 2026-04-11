"""
Finite difference gradient check for wave32 (RDNA3) backward kernels.

Compares the analytical gradients from the CUDA backward pass against
numerical (finite difference) gradients.

Usage:
    python3 tests/test_wave32_gradients.py
"""

import torch
import math

device = torch.device("cuda:0")


def create_scene(N=20, width=32, height=32, focal=50.0, seed=42):
    """Create a simple scene with gaussians in front of the camera."""
    torch.manual_seed(seed)

    means = torch.zeros((N, 3), device=device)
    means[:, 0] = (torch.rand(N, device=device) - 0.5) * 2.0
    means[:, 1] = (torch.rand(N, device=device) - 0.5) * 2.0
    means[:, 2] = 2.0 + torch.rand(N, device=device) * 2.0

    quats = torch.randn((N, 4), device=device)
    quats = quats / quats.norm(dim=-1, keepdim=True)
    scales = torch.rand((N, 3), device=device) * 0.2 + 0.05
    opacities = torch.rand((N,), device=device) * 0.3 + 0.7
    colors = torch.rand((N, 3), device=device)

    viewmats = torch.eye(4, device=device)[None, :, :]
    Ks = torch.tensor(
        [[focal, 0.0, width / 2.0], [0.0, focal, height / 2.0], [0.0, 0.0, 1.0]],
        device=device,
    )[None, :, :]

    return means, quats, scales, opacities, colors, viewmats, Ks, width, height


def do_rasterization(
    means, quats, scales, opacities, colors, viewmats, Ks, width, height, tile_size=16
):
    from gsplat.rendering import rasterization

    return rasterization(
        means=means,
        quats=quats,
        scales=scales,
        opacities=opacities,
        colors=colors,
        viewmats=viewmats,
        Ks=Ks,
        width=width,
        height=height,
        tile_size=tile_size,
        packed=False,
    )


def finite_diff_check(param_name, param, all_params, eps=5e-4, max_elements=20):
    """
    Compare analytical gradient vs finite difference gradient.
    Returns max relative error and status.
    """
    from gsplat.rendering import rasterization

    means, quats, scales, opacities, colors, viewmats, Ks, width, height = all_params

    # Compute analytical gradient
    kw = dict(
        quats=quats,
        scales=scales,
        opacities=opacities,
        colors=colors,
        viewmats=viewmats,
        Ks=Ks,
        width=width,
        height=height,
        tile_size=16,
        packed=False,
    )

    if param_name == "colors":
        colors_req = param.clone().detach().requires_grad_(True)
        kw["colors"] = colors_req
        kw.pop("opacities", None)
        r, a, m = rasterization(means=means, opacities=opacities, **kw)
        loss = r.sum()
        loss.backward()
        analytical_grad = colors_req.grad.clone()
    elif param_name == "opacities":
        opac_req = param.clone().detach().requires_grad_(True)
        kw["opacities"] = opac_req
        r, a, m = rasterization(means=means, **kw)
        loss = r.sum()
        loss.backward()
        analytical_grad = opac_req.grad.clone()
    elif param_name == "means":
        means_req = param.clone().detach().requires_grad_(True)
        r, a, m = rasterization(means=means_req, **kw)
        loss = r.sum()
        loss.backward()
        analytical_grad = means_req.grad.clone()
    else:
        raise ValueError(f"Unknown param: {param_name}")

    # Compute finite difference gradient for a subset of elements
    flat_param = param.detach().clone().flatten()
    n_check = min(max_elements, flat_param.numel())

    indices = torch.linspace(0, flat_param.numel() - 1, n_check).long()

    errors = []

    for idx in indices:
        idx = idx.item()
        orig = flat_param[idx].item()

        # f(x + eps)
        flat_param[idx] = orig + eps
        perturbed = flat_param.reshape(param.shape)
        r_p, _, _ = do_rasterization(
            means if param_name != "means" else perturbed,
            quats,
            scales,
            opacities if param_name != "opacities" else perturbed,
            colors if param_name != "colors" else perturbed,
            viewmats,
            Ks,
            width,
            height,
        )
        loss_plus = r_p.sum()

        # f(x - eps)
        flat_param[idx] = orig - eps
        perturbed = flat_param.reshape(param.shape)
        r_m, _, _ = do_rasterization(
            means if param_name != "means" else perturbed,
            quats,
            scales,
            opacities if param_name != "opacities" else perturbed,
            colors if param_name != "colors" else perturbed,
            viewmats,
            Ks,
            width,
            height,
        )
        loss_minus = r_m.sum()

        # Restore
        flat_param[idx] = orig

        fd = (loss_plus - loss_minus).item() / (2 * eps)
        ana = analytical_grad.flatten()[idx].item()

        if abs(fd) > 1e-6 or abs(ana) > 1e-6:
            denom = max(abs(fd), abs(ana))
            rel_err = abs(fd - ana) / denom
            errors.append((rel_err, idx, fd, ana))

    if not errors:
        return 0.0, 0, "NO NON-ZERO GRADIENTS"

    max_err = max(e[0] for e in errors)
    mean_err = sum(e[0] for e in errors) / len(errors)

    # Print worst mismatches
    errors.sort(key=lambda x: -x[0])
    print(f"  Top 5 worst elements:")
    for rel_err, idx, fd, ana in errors[:5]:
        print(f"    idx={idx}: fd={fd:.6f}, ana={ana:.6f}, rel_err={rel_err:.6f}")

    status = "FAIL" if max_err > 0.1 else "PASS"
    if max_err < 0.01:
        status = "PASS (excellent)"
    elif max_err < 0.05:
        status = "PASS (good)"

    return max_err, len(errors), status


def main():
    from gsplat.rendering import rasterization

    print("=" * 70)
    print("WAVE32 GRADIENT VALIDATION TEST")
    print("=" * 70)
    print(f"Device: {torch.cuda.get_device_name(0)}")
    print(f"Warp size: {torch.cuda.get_device_properties(0).warp_size}")
    print()

    scene = create_scene(N=20, width=32, height=32, focal=50.0, seed=42)
    means, quats, scales, opacities, colors, viewmats, Ks, width, height = scene

    all_passed = True

    for param_name, param in [
        ("colors", colors),
        ("opacities", opacities),
        ("means", means),
    ]:
        print(f"--- {param_name} gradient check (tile_size=16) ---")
        max_err, n_checked, status = finite_diff_check(
            param_name,
            param,
            scene,
            eps=5e-4,
            max_elements=20,
        )
        print(f"  Elements checked: {n_checked}")
        print(f"  Max relative error: {max_err:.6f}")
        print(f"  Status: {status}")
        print()
        if "FAIL" in status:
            all_passed = False

    # Test tile_size=8 (triggers bs64 kernel path)
    print("--- tile_size=8 test (bs64 kernel) ---")
    scene8 = create_scene(N=20, width=16, height=16, focal=30.0, seed=99)
    m8, q8, s8, o8, c8, v8, k8, w8, h8 = scene8
    try:
        m = m8.clone().requires_grad_(True)
        c = c8.clone().requires_grad_(True)
        o = o8.clone().requires_grad_(True)
        r, a, meta = do_rasterization(m, q8, s8, o, c, v8, k8, w8, h8, tile_size=8)
        loss = r.sum()
        loss.backward()
        has_nan_m = torch.isnan(m.grad).any().item()
        has_nan_c = torch.isnan(c.grad).any().item()
        has_nan_o = torch.isnan(o.grad).any().item()
        print(
            f"  render_sum={r.sum():.4f}, alpha_coverage={((a > 0.01).float().mean().item()):.2%}"
        )
        print(
            f"  grad NaN: means={has_nan_m}, colors={has_nan_c}, opacities={has_nan_o}"
        )
        if has_nan_m or has_nan_c or has_nan_o:
            all_passed = False
            print("  Status: FAIL (NaN in gradients)")
        else:
            print("  Status: PASS (no NaN)")
    except Exception as e:
        print(f"  ERROR: {e}")
        all_passed = False
    print()

    # Gradient consistency (two runs should be identical)
    print("--- Gradient consistency (determinism) ---")
    grads_runs = []
    for run in range(2):
        m = means.clone().requires_grad_(True)
        c = colors.clone().requires_grad_(True)
        o = opacities.clone().requires_grad_(True)
        r, a, meta = do_rasterization(
            m, quats, scales, o, c, viewmats, Ks, width, height
        )
        loss = r.sum()
        loss.backward()
        grads_runs.append(
            {
                "means": m.grad.clone(),
                "colors": c.grad.clone(),
                "opacities": o.grad.clone(),
            }
        )

    for name in ["means", "colors", "opacities"]:
        diff = (grads_runs[0][name] - grads_runs[1][name]).abs().max().item()
        # atomicAdd accumulation order is non-deterministic; float32 addition
        # is not associative, so ULP-level differences (~1e-5) are expected.
        status = "PASS" if diff < 1e-4 else "FAIL"
        if diff >= 1e-4:
            all_passed = False
        print(f"  {name}: max diff = {diff:.10f} [{status}]")

    print()
    print("=" * 70)
    if all_passed:
        print("ALL TESTS PASSED")
    else:
        print("SOME TESTS FAILED - gradients may be incorrect!")
    print("=" * 70)

    return all_passed


if __name__ == "__main__":
    import sys

    passed = main()
    sys.exit(0 if passed else 1)
