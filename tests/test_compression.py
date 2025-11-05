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

@pytest.mark.skip()
@pytest.mark.skipif(not torch.cuda.is_available(), reason="No CUDA device")
def test_png_compression():
    from gsplat.compression import PngCompression

    torch.manual_seed(42)

    # Prepare Gaussians
    N = 100000
    splats = torch.nn.ParameterDict(
        {
            "means": torch.randn(N, 3),
            "scales": torch.randn(N, 3),
            "quats": torch.randn(N, 4),
            "opacities": torch.randn(N),
            "sh0": torch.randn(N, 1, 3),
            "shN": torch.randn(N, 24, 3),
            "features": torch.randn(N, 128),
        }
    ).to(device)
    compress_dir = "/tmp/gsplat/compression"

    compression_method = PngCompression()
    # run compression and save the compressed files to compress_dir
    compression_method.compress(compress_dir, splats)
    # decompress the compressed files
    splats_c = compression_method.decompress(compress_dir)


if __name__ == "__main__":
    test_png_compression()
