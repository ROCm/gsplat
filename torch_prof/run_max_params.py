#!/usr/bin/env python3
"""Run profiling with maximum parameters to stress test the system."""

import subprocess
import os


def run_max_main_profiling():
    """Run main.py with maximum parameters."""
    print("Running main.py with maximum parameters...")
    
    cmd = [
        "python", "profiling/main.py",
        "--batch_size", "8",
        "--scene_grid", "21", 
        "--channels", "32",
        "--resolution", "4k",
        "--repeats", "5"
    ]
    
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = "0"
    
    print(f"Command: {' '.join(cmd)}")
    
    try:
        result = subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=1800)  # 30 min timeout
        print("✓ Main profiling completed")
        return True
    except subprocess.TimeoutExpired:
        print("✗ Main profiling timed out")
        return False
    except Exception as e:
        print(f"✗ Error running main profiling: {e}")
        return False


def run_max_test_profiling():
    """Run tests with maximum parameters."""
    
    max_tests = [
        "tests/test_basic.py::test_rasterize_to_pixels[batch_dims2-128]",
        "tests/test_rasterization.py::test_rasterization[batch_dims2-True-RGB+D-3-True]", 
        "tests/test_2dgs.py::test_rasterize_to_pixels_2dgs[batch_dims2-31]",
        "tests/test_strategy.py::test_strategy",
    ]
    
    env = os.environ.copy()
    env["ENABLE_PROFILER"] = "1" 
    env["CUDA_VISIBLE_DEVICES"] = "0"
    
    for test in max_tests:
        print(f"\nRunning: {test}")
        
        cmd = ["pytest", "-v", "-s", test]
        
        try:
            result = subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=600)  # 10 min timeout
            if result.returncode == 0:
                print(f"✓ {test} completed")
            else:
                print(f"✗ {test} failed")
                print(result.stderr[:500])
        except subprocess.TimeoutExpired:
            print(f"✗ {test} timed out")
        except Exception as e:
            print(f"✗ Error running {test}: {e}")


def run_kernel_profiling():
    """Run kernel profiling."""
    print("\nRunning kernel profiling...")
    
    cmd = ["python", "torch_prof/profile_kernels.py"]
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = "0"
    
    try:
        result = subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=600)
        if result.returncode == 0:
            print("✓ Kernel profiling completed")
        else:
            print("✗ Kernel profiling failed")
            print(result.stderr[:500])
    except Exception as e:
        print(f"✗ Error running kernel profiling: {e}")


def main():
    print("="*80)
    print("RUNNING PROFILING WITH MAXIMUM PARAMETERS")
    print("="*80)
    
    # Change to gsplat directory
    os.chdir("/home/AMD/chanshar/workspace/gsplat")
    
    # Run main profiling with max params
    run_max_main_profiling()
    
    # Run tests with max params  
    run_max_test_profiling()
    
    # Run kernel profiling
    run_kernel_profiling()
    
    print("\n" + "="*80)
    print("PROFILING COMPLETE")
    print("="*80)
    print("Results in: torch_prof/logs/")
    print("\nAnalyze with:")
    print("  python torch_prof/extract_kernel_times.py")
    print("  python torch_prof/use_tracelens.py --all")


if __name__ == "__main__":
    main()