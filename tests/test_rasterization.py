"""Tests for the functions in the CUDA extension.

Usage:
```bash
pytest <THIS_PY_FILE> -s
```
"""

from typing import Optional, Tuple

import pytest
import torch



import os
import re

# --- Pytest Fixture for Profiling ---
@pytest.fixture(autouse=True)
def profile_test_with_torch(request):
    """
    An autouse fixture that profiles each test function using torch.profiler.
    Profiling is only enabled if the environment variable ENABLE_PROFILER is set to "1".
    """
    if os.getenv("ENABLE_PROFILER") != "1":
        yield
        return
    
    node_id = request.node.nodeid
    sanitized_node_id = re.sub(r'[:\[\]\-/]', '_', node_id.split("::")[-1])
    
    activities = [torch.profiler.ProfilerActivity.CPU]
    if torch.cuda.is_available():
        activities.append(torch.profiler.ProfilerActivity.CUDA)
    
    log_dir = f"./torch_prof/logs/tests/{sanitized_node_id}"
    print(f"\n[Profiler] Enabled for {node_id}.")
    print(f"[Profiler] Trace will be saved to '{log_dir}'")
    
    with torch.profiler.profile(
        activities=activities,
        on_trace_ready=torch.profiler.tensorboard_trace_handler(log_dir),
        record_shapes=True,
        profile_memory=True,
        with_stack=True
    ) as prof:
        yield
    
    print(f"\n[Profiler] Results for {node_id}:")
    print(prof.key_averages().table(sort_by="cuda_time_total" if torch.cuda.is_available() else "cpu_time_total", row_limit=15))


device = torch.device("cuda:0")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="No CUDA device")
@pytest.mark.parametrize("per_view_color", [True, False])
@pytest.mark.parametrize("sh_degree", [None, 3])
@pytest.mark.parametrize("render_mode", ["RGB", "RGB+D", "D"])
@pytest.mark.parametrize("packed", [True, False])
@pytest.mark.parametrize("batch_dims", [(), (2,), (1, 2)])
def test_rasterization(
    per_view_color: bool,
    sh_degree: Optional[int],
    render_mode: str,
    packed: bool,
    batch_dims: Tuple[int, ...],
):
    from gsplat.rendering import _rasterization, rasterization

    torch.manual_seed(42)

    C, N = 3, 10_000
    means = torch.rand(batch_dims + (N, 3), device=device)
    quats = torch.randn(batch_dims + (N, 4), device=device)
    scales = torch.rand(batch_dims + (N, 3), device=device)
    opacities = torch.rand(batch_dims + (N,), device=device)
    if per_view_color:
        if sh_degree is None:
            colors = torch.rand(batch_dims + (C, N, 3), device=device)
        else:
            colors = torch.rand(
                batch_dims + (C, N, (sh_degree + 1) ** 2, 3), device=device
            )
    else:
        if sh_degree is None:
            colors = torch.rand(batch_dims + (N, 3), device=device)
        else:
            colors = torch.rand(
                batch_dims + (N, (sh_degree + 1) ** 2, 3), device=device
            )

    width, height = 300, 200
    focal = 300.0
    Ks = torch.tensor(
        [[focal, 0.0, width / 2.0], [0.0, focal, height / 2.0], [0.0, 0.0, 1.0]],
        device=device,
    ).expand(batch_dims + (C, -1, -1))
    viewmats = torch.eye(4, device=device).expand(batch_dims + (C, -1, -1))

    renders, alphas, meta = rasterization(
        means=means,
        quats=quats,
        scales=scales,
        opacities=opacities,
        colors=colors,
        viewmats=viewmats,
        Ks=Ks,
        width=width,
        height=height,
        sh_degree=sh_degree,
        render_mode=render_mode,
        packed=packed,
    )

    if render_mode == "D":
        assert renders.shape == batch_dims + (C, height, width, 1)
    elif render_mode == "RGB":
        assert renders.shape == batch_dims + (C, height, width, 3)
    elif render_mode == "RGB+D":
        assert renders.shape == batch_dims + (C, height, width, 4)

    _renders, _alphas, _meta = _rasterization(
        means=means,
        quats=quats,
        scales=scales,
        opacities=opacities,
        colors=colors,
        viewmats=viewmats,
        Ks=Ks,
        width=width,
        height=height,
        sh_degree=sh_degree,
        render_mode=render_mode,
    )
    torch.testing.assert_close(renders, _renders, rtol=1e-4, atol=1e-4)
    torch.testing.assert_close(alphas, _alphas, rtol=1e-4, atol=1e-4)
