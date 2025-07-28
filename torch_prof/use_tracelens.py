#!/usr/bin/env python3
"""Use TraceLens to generate Excel reports from gsplat profiling traces."""

import os
import sys
import glob
import subprocess
from pathlib import Path
import argparse


def find_trace_files(log_dirs=None):
    """Find all trace files in specified directories."""
    if log_dirs is None:
        log_dirs = ["torch_prof/logs"]
    
    trace_files = []
    for log_dir in log_dirs:
        if os.path.exists(log_dir):
            pattern = f"{log_dir}/**/*.pt.trace.json"
            trace_files.extend(glob.glob(pattern, recursive=True))
    
    return trace_files


def generate_tracelens_report(trace_file, output_dir="torch_prof/tracelens_reports"):
    """Generate TraceLens report for a single trace file."""
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate output filename
    trace_name = Path(trace_file).parent.name
    output_file = os.path.join(output_dir, f"{trace_name}.xlsx")
    
    # TraceLens path
    tracelens_path = "./TraceLens-main/examples"
    tracelens_script = os.path.join(tracelens_path, "generate_perf_report.py")
    
    if not os.path.exists(tracelens_script):
        print(f"Error: TraceLens script not found at {tracelens_script}")
        return False
    
    # Convert to absolute paths
    abs_trace_file = os.path.abspath(trace_file)
    abs_output_file = os.path.abspath(output_file)
    
    # Run TraceLens
    cmd = [
        "python3", 
        tracelens_script,
        "--profile_json_path", abs_trace_file,
        "--output_xlsx_path", abs_output_file
    ]
    
    print(f"Generating report for: {trace_name}")
    print(f"Command: {' '.join(cmd)}")
    
    try:
        # Change to TraceLens directory
        original_cwd = os.getcwd()
        os.chdir(tracelens_path)
        
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        
        # Change back
        os.chdir(original_cwd)
        
        if result.returncode == 0:
            print(f"✓ Report generated: {output_file}")
            return True
        else:
            print(f"✗ Error generating report for {trace_name}")
            print(f"STDERR: {result.stderr}")
            return False
            
    except subprocess.TimeoutExpired:
        print(f"✗ Timeout generating report for {trace_name}")
        return False
    except Exception as e:
        print(f"✗ Exception generating report for {trace_name}: {e}")
        return False


def generate_all_reports():
    """Generate TraceLens reports for all trace files."""
    
    # Find all trace files
    trace_files = find_trace_files()
    
    if not trace_files:
        print("No trace files found!")
        print("Make sure you have run profiling first:")
        print("  ENABLE_PROFILER=1 CUDA_VISIBLE_DEVICES=0 pytest -v -s tests/test_basic.py")
        return
    
    print(f"Found {len(trace_files)} trace files")
    
    # Process each trace file
    successful = 0
    failed = 0
    
    for trace_file in trace_files:
        print(f"\n{'='*60}")
        print(f"Processing: {trace_file}")
        
        if generate_tracelens_report(trace_file):
            successful += 1
        else:
            failed += 1
    
    print(f"\n{'='*60}")
    print("TRACELENS REPORT GENERATION COMPLETE")
    print(f"{'='*60}")
    print(f"Successful: {successful}")
    print(f"Failed: {failed}")
    print(f"Reports saved in: torch_prof/tracelens_reports/")
    
    # List generated reports
    report_dir = "torch_prof/tracelens_reports"
    if os.path.exists(report_dir):
        reports = glob.glob(f"{report_dir}/*.xlsx")
        if reports:
            print(f"\nGenerated Excel reports:")
            for report in sorted(reports):
                size_mb = os.path.getsize(report) / (1024 * 1024)
                print(f"  - {os.path.basename(report)} ({size_mb:.1f} MB)")


def generate_single_report(trace_path):
    """Generate TraceLens report for a single trace file."""
    
    if not os.path.exists(trace_path):
        print(f"Error: Trace file not found: {trace_path}")
        return
    
    success = generate_tracelens_report(trace_path)
    
    if success:
        trace_name = Path(trace_path).parent.name
        output_file = f"torch_prof/tracelens_reports/{trace_name}.xlsx"
        print(f"\n✓ Report generated successfully!")
        print(f"Open with: libreoffice {output_file}")
        print(f"Or copy to local machine for Excel viewing")
    else:
        print(f"\n✗ Failed to generate report")


def list_available_traces():
    """List all available trace files."""
    trace_files = find_trace_files()
    
    if not trace_files:
        print("No trace files found!")
        return
    
    print(f"Available trace files ({len(trace_files)}):")
    print("="*80)
    
    # Group by category
    categories = {}
    for trace_file in trace_files:
        path_parts = Path(trace_file).parts
        if 'tests' in path_parts:
            category = 'Tests'
        elif 'kernels' in path_parts:
            category = 'Kernels'
        elif 'end_to_end' in path_parts:
            category = 'End-to-End'
        else:
            category = 'Other'
        
        if category not in categories:
            categories[category] = []
        
        size_mb = os.path.getsize(trace_file) / (1024 * 1024)
        categories[category].append((Path(trace_file).parent.name, trace_file, size_mb))
    
    for category, items in categories.items():
        print(f"\n{category}:")
        for name, path, size in sorted(items):
            print(f"  - {name:<40} ({size:.1f} MB)")
            print(f"    {path}")


def main():
    parser = argparse.ArgumentParser(description="Generate TraceLens reports from gsplat profiling")
    parser.add_argument("--all", action="store_true", help="Generate reports for all trace files")
    parser.add_argument("--trace", help="Generate report for specific trace file")
    parser.add_argument("--list", action="store_true", help="List available trace files")
    
    args = parser.parse_args()
    
    if args.list:
        list_available_traces()
    elif args.trace:
        generate_single_report(args.trace)
    elif args.all:
        generate_all_reports()
    else:
        print("Usage examples:")
        print("  # List available traces")
        print("  python torch_prof/use_tracelens.py --list")
        print("")
        print("  # Generate reports for all traces")
        print("  python torch_prof/use_tracelens.py --all")
        print("")
        print("  # Generate report for specific trace")
        print("  python torch_prof/use_tracelens.py --trace logs/test_rasterize_to_pixels_batch_dims2_128_/*.pt.trace.json")


if __name__ == "__main__":
    main()