"""Tests for the functions in the CUDA extension.

Usage:
```bash
pytest <THIS_PY_FILE> -s
```
"""

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
def test_strategy():
    from gsplat.rendering import rasterization
    from gsplat.strategy import DefaultStrategy, MCMCStrategy

    torch.manual_seed(42)

    # Prepare Gaussians
    N = 100
    params = torch.nn.ParameterDict(
        {
            "means": torch.randn(N, 3),
            "scales": torch.rand(N, 3),
            "quats": torch.randn(N, 4),
            "opacities": torch.rand(N),
            "colors": torch.rand(N, 3),
        }
    ).to(device)
    optimizers = {k: torch.optim.Adam([v], lr=1e-3) for k, v in params.items()}

    # A dummy rendering call
    render_colors, render_alphas, info = rasterization(
        means=params["means"],
        quats=params["quats"],  # F.normalize is fused into the kernel
        scales=torch.exp(params["scales"]),
        opacities=torch.sigmoid(params["opacities"]),
        colors=params["colors"],
        viewmats=torch.eye(4).unsqueeze(0).to(device),
        Ks=torch.eye(3).unsqueeze(0).to(device),
        width=10,
        height=10,
        packed=False,
    )

    # Test DefaultStrategy
    strategy = DefaultStrategy(verbose=True)
    strategy.check_sanity(params, optimizers)
    state = strategy.initialize_state()
    strategy.step_pre_backward(params, optimizers, state, step=600, info=info)
    render_colors.mean().backward(retain_graph=True)
    strategy.step_post_backward(params, optimizers, state, step=600, info=info)

    # Test MCMCStrategy
    strategy = MCMCStrategy(verbose=True)
    strategy.check_sanity(params, optimizers)
    state = strategy.initialize_state()
    render_colors.mean().backward(retain_graph=True)
    strategy.step_post_backward(params, optimizers, state, step=600, info=info, lr=1e-3)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="No CUDA device")
def test_strategy_requires_grad():
    from gsplat.rendering import rasterization
    from gsplat.strategy import DefaultStrategy, MCMCStrategy

    def assert_consistent_sizes(params):
        sizes = [v.shape[0] for v in params.values()]
        assert all([s == sizes[0] for s in sizes])

    torch.manual_seed(42)

    # Prepare Gaussians
    N = 100
    params = torch.nn.ParameterDict(
        {
            "means": torch.randn(N, 3),
            "scales": torch.rand(N, 3),
            "quats": torch.randn(N, 4),
            "opacities": torch.rand(N),
            "colors": torch.rand(N, 3),
            "non_trainable_features": torch.rand(N, 3),
        }
    ).to(device)
    params["non_trainable_features"].requires_grad = False
    requires_grad_map = {k: v.requires_grad for k, v in params.items()}
    optimizers = {
        k: torch.optim.Adam([v], lr=1e-3) for k, v in params.items() if v.requires_grad
    }

    # A dummy rendering call
    render_colors, render_alphas, info = rasterization(
        means=params["means"],
        quats=params["quats"],  # F.normalize is fused into the kernel
        scales=torch.exp(params["scales"]),
        opacities=torch.sigmoid(params["opacities"]),
        colors=params["colors"],
        viewmats=torch.eye(4).unsqueeze(0).to(device),
        Ks=torch.eye(3).unsqueeze(0).to(device),
        width=10,
        height=10,
        packed=False,
    )

    # Test DefaultStrategy
    strategy = DefaultStrategy(verbose=True)
    strategy.check_sanity(params, optimizers)
    state = strategy.initialize_state()
    strategy.step_pre_backward(params, optimizers, state, step=600, info=info)
    render_colors.mean().backward(retain_graph=True)
    strategy.step_post_backward(params, optimizers, state, step=600, info=info)
    for k, v in params.items():
        assert v.requires_grad == requires_grad_map[k]
    assert params["non_trainable_features"].grad is None
    assert_consistent_sizes(params)
    # Test MCMCStrategy
    strategy = MCMCStrategy(verbose=True)
    strategy.check_sanity(params, optimizers)
    state = strategy.initialize_state()
    render_colors.mean().backward(retain_graph=True)
    strategy.step_post_backward(params, optimizers, state, step=600, info=info, lr=1e-3)
    assert params["non_trainable_features"].grad is None
    for k, v in params.items():
        assert v.requires_grad == requires_grad_map[k]
    assert_consistent_sizes(params)


if __name__ == "__main__":
    test_strategy()
    test_strategy_requires_grad()
