#!/usr/bin/env python3
"""Automated profiling script for all gsplat tests and benchmarks."""

import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import subprocess
import argparse
from pathlib import Path
import json
from datetime import datetime
import importlib.util


def run_pytest_with_profiling(test_file: str, specific_test: str = None):
    """Run pytest with profiling enabled for a specific test file."""
    env = os.environ.copy()
    env["ENABLE_PROFILER"] = "1"
    
    cmd = ["pytest", "-v", test_file]
    if specific_test:
        cmd.extend(["-k", specific_test])
    
    print(f"\n{'='*60}")
    print(f"Running profiled tests for: {test_file}")
    if specific_test:
        print(f"Specific test: {specific_test}")
    print(f"{'='*60}\n")
    
    print(f"Running command: {' '.join(cmd)}")
    result = subprocess.run(cmd, env=env, capture_output=True, text=True)
    
    return {
        "file": test_file,
        "specific_test": specific_test,
        "return_code": result.returncode,
        "stdout": result.stdout,
        "stderr": result.stderr,
        "success": result.returncode == 0
    }


def profile_main_script():
    """Profile the main.py script in profiling directory."""
    from torch_prof.profiler import profile_code
    
    # Add the parent directory to path to import main
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
    
    # Import the worker function from main.py
    spec = importlib.util.spec_from_file_location("main", "../profiling/main.py")
    main_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(main_module)
    
    # Create mock args for profiling
    class Args:
        resolution = ["360p"]
        scene_size = [1]
        batch_sizes = [1]
        channels = [3]
        repeats = 5
        memory_history = False
    
    args = Args()
    
    print(f"\n{'='*60}")
    print(f"Profiling main.py benchmarks")
    print(f"{'='*60}\n")
    
    # Profile the worker function
    with profile_code("main_benchmark", group="profiling"):
        # Run single worker instance
        main_module.worker(0, 0, 1, args)
    
    return {"file": "profiling/main.py", "success": True}


def run_trainer_with_profiling(trainer_script: str, trainer_args: list = None):
    """Run training script with profiling enabled."""
    env = os.environ.copy()
    env["ENABLE_PROFILER"] = "1"
    
    # Base command
    cmd = ["python", trainer_script]
    
    # Add training arguments
    if trainer_args:
        cmd.extend(trainer_args)
    
    print(f"\n{'='*60}")
    print(f"Running profiled training: {trainer_script}")
    print(f"Arguments: {' '.join(trainer_args) if trainer_args else 'None'}")
    print(f"{'='*60}\n")
    
    print(f"Running command: {' '.join(cmd)}")
    result = subprocess.run(cmd, env=env, capture_output=True, text=True)
    
    return {
        "file": trainer_script,
        "arguments": trainer_args,
        "return_code": result.returncode,
        "stdout": result.stdout,
        "stderr": result.stderr,
        "success": result.returncode == 0
    }


def inject_profiler_to_tests():
    """Inject profiler fixture into test files that don't have it."""
    test_files = list(Path("tests").glob("test_*.py"))
    
    profiler_code = '''

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
    print(f"\\n[Profiler] Enabled for {node_id}.")
    print(f"[Profiler] Trace will be saved to '{log_dir}'")
    
    with torch.profiler.profile(
        activities=activities,
        on_trace_ready=torch.profiler.tensorboard_trace_handler(log_dir),
        record_shapes=True,
        profile_memory=True,
        with_stack=True
    ) as prof:
        yield
    
    print(f"\\n[Profiler] Results for {node_id}:")
    print(prof.key_averages().table(sort_by="cuda_time_total" if torch.cuda.is_available() else "cpu_time_total", row_limit=15))

'''
    
    injected_files = []
    
    for test_file in test_files:
        if test_file.name == "test_basic.py":
            continue  # Skip as it already has profiler
        
        content = test_file.read_text()
        
        # Check if profiler already exists
        if "profile_test_with_torch" in content:
            continue
        
        # Find the right place to inject (after imports but before any code)
        lines = content.split('\n')
        import_end = 0
        in_docstring = False
        
        for i, line in enumerate(lines):
            # Handle multi-line docstrings
            if '"""' in line or "'''" in line:
                in_docstring = not in_docstring
                continue
            
            if in_docstring:
                continue
                
            # Skip empty lines, imports, and comments
            if not line.strip() or line.startswith(('import ', 'from ', '#')):
                continue
            
            # Found first non-import line
            import_end = i
            break
        
        # Make sure torch and pytest are imported
        torch_imported = 'import torch' in content or 'from torch' in content
        pytest_imported = 'import pytest' in content or 'from pytest' in content
        
        additional_imports = []
        if not torch_imported:
            additional_imports.append('import torch')
        if not pytest_imported:
            additional_imports.append('import pytest')
        
        # Inject imports if needed
        if additional_imports:
            lines.insert(import_end, '\n'.join(additional_imports))
            import_end += len(additional_imports)
        
        # Inject the profiler code
        lines.insert(import_end, profiler_code)
        
        # Write back
        test_file.write_text('\n'.join(lines))
        injected_files.append(str(test_file))
    
    return injected_files


def main():
    parser = argparse.ArgumentParser(description="Profile all gsplat tests and benchmarks")
    parser.add_argument("--inject-profiler", action="store_true", 
                       help="Inject profiler fixture into test files that don't have it")
    parser.add_argument("--tests", nargs="*", 
                       help="Specific test files to profile (e.g., test_basic.py)")
    parser.add_argument("--specific-test", 
                       help="Run only a specific test function")
    parser.add_argument("--include-main", action="store_true",
                       help="Also profile the main.py benchmark script")
    parser.add_argument("--trainers", nargs="*", choices=["simple_trainer", "simple_trainer_2dgs"],
                       help="Profile training scripts (simple_trainer, simple_trainer_2dgs)")
    parser.add_argument("--trainer-data-dir", 
                       help="Data directory for training (optional, uses trainer defaults if not specified)")
    parser.add_argument("--summary-file", default="torch_prof/profiling_summary.json",
                       help="Output file for profiling summary")
    
    args = parser.parse_args()
    
    results = []
    
    # Inject profiler if requested
    if args.inject_profiler:
        injected = inject_profiler_to_tests()
        if injected:
            print(f"Injected profiler into: {', '.join(injected)}")
        else:
            print("No files needed profiler injection")
    
    # Determine which test files to run
    if args.tests:
        test_files = [f"tests/{t}" if not t.startswith("tests/") else t for t in args.tests]
        
        # Run profiling on each test file
        for test_file in test_files:
            if not os.path.exists(test_file):
                print(f"Warning: {test_file} not found, skipping...")
                continue
            
            result = run_pytest_with_profiling(test_file, args.specific_test)
            results.append(result)
            
            if not result["success"]:
                print(f"\nError running {test_file}:")
                print(f"Return code: {result['return_code']}")
                print(f"STDERR:\n{result['stderr']}")
                print(f"STDOUT:\n{result['stdout'][:500]}...")  # First 500 chars
    
    # Profile main.py if requested
    if args.include_main:
        try:
            main_result = profile_main_script()
            results.append(main_result)
        except Exception as e:
            print(f"\nError profiling main.py: {e}")
            results.append({
                "file": "profiling/main.py",
                "success": False,
                "error": str(e)
            })
    
    # Profile training scripts if requested
    if args.trainers:
        # Build trainer arguments - only add data_dir if specified
        trainer_args = []
        if args.trainer_data_dir:
            trainer_args.append(f"data_dir={args.trainer_data_dir}")
        
        for trainer in args.trainers:
            trainer_script = f"examples/{trainer}.py"
            if not os.path.exists(trainer_script):
                print(f"Warning: {trainer_script} not found, skipping...")
                continue
            
            try:
                trainer_result = run_trainer_with_profiling(trainer_script, trainer_args)
                results.append(trainer_result)
                
                if not trainer_result["success"]:
                    print(f"\nError running {trainer_script}:")
                    print(f"Return code: {trainer_result['return_code']}")
                    print(f"STDERR:\n{trainer_result['stderr']}")
                    print(f"STDOUT:\n{trainer_result['stdout'][:500]}...")  # First 500 chars
                    
            except Exception as e:
                print(f"\nError profiling {trainer_script}: {e}")
                results.append({
                    "file": trainer_script,
                    "success": False,
                    "error": str(e)
                })
    
    # Generate summary
    summary = {
        "timestamp": datetime.now().isoformat(),
        "total_files": len(results),
        "successful": sum(1 for r in results if r["success"]),
        "failed": sum(1 for r in results if not r["success"]),
        "results": results
    }
    
    # Save summary
    os.makedirs(os.path.dirname(args.summary_file), exist_ok=True)
    with open(args.summary_file, "w") as f:
        json.dump(summary, f, indent=2)
    
    # Print summary
    print(f"\n{'='*60}")
    print("PROFILING SUMMARY")
    print(f"{'='*60}")
    print(f"Total files profiled: {summary['total_files']}")
    print(f"Successful: {summary['successful']}")
    print(f"Failed: {summary['failed']}")
    print(f"\nSummary saved to: {args.summary_file}")
    print(f"\nView profiling results with: tensorboard --logdir=torch_prof/logs")
    
    return 0 if summary['failed'] == 0 else 1


if __name__ == "__main__":
    sys.exit(main())