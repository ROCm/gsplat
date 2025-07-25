#!/usr/bin/env python3
"""Extract total_direct_kernel_time_ms from all trace files."""

import json
import os
import glob
from pathlib import Path
import csv


def extract_kernel_time_from_trace(trace_file):
    """Extract total_direct_kernel_time_ms from a trace file."""
    try:
        with open(trace_file, 'r') as f:
            data = json.load(f)
        
        # Look for kernel_launchers_summary
        if 'kernel_launchers_summary' in data:
            kernel_summary = data['kernel_launchers_summary']
            total_time = kernel_summary.get('total_direct_kernel_time_ms', 0)
            return total_time
        else:
            # Sometimes it might be nested differently, search recursively
            def find_kernel_time(obj):
                if isinstance(obj, dict):
                    if 'total_direct_kernel_time_ms' in obj:
                        return obj['total_direct_kernel_time_ms']
                    for v in obj.values():
                        result = find_kernel_time(v)
                        if result is not None:
                            return result
                elif isinstance(obj, list):
                    for item in obj:
                        result = find_kernel_time(item)
                        if result is not None:
                            return result
                return None
            
            result = find_kernel_time(data)
            return result if result is not None else 0
            
    except Exception as e:
        print(f"Error reading {trace_file}: {e}")
        return None


def analyze_all_kernel_times():
    """Analyze kernel times from all trace files."""
    
    # Find all trace files
    trace_files = []
    for pattern in ["torch_prof/logs/**/*.pt.trace.json"]:
        trace_files.extend(glob.glob(pattern, recursive=True))
    
    if not trace_files:
        print("No trace files found!")
        return
    
    print(f"Analyzing {len(trace_files)} trace files...\n")
    
    results = []
    
    for trace_file in trace_files:
        # Extract test/kernel name
        path_parts = Path(trace_file).parts
        if 'tests' in path_parts:
            category = 'Test'
            name = Path(trace_file).parent.name
        elif 'kernels' in path_parts:
            category = 'Kernel'
            name = Path(trace_file).parent.name
        else:
            category = 'Other'
            name = Path(trace_file).parent.name
        
        # Extract kernel time
        kernel_time = extract_kernel_time_from_trace(trace_file)
        
        if kernel_time is not None:
            results.append({
                'name': name,
                'category': category,
                'total_direct_kernel_time_ms': kernel_time,
                'trace_file': trace_file
            })
    
    # Sort by kernel time (descending)
    results.sort(key=lambda x: x['total_direct_kernel_time_ms'], reverse=True)
    
    # Print summary
    print("="*80)
    print("KERNEL TIME SUMMARY (total_direct_kernel_time_ms)")
    print("="*80)
    print(f"{'Name':<50} {'Category':<10} {'Kernel Time (ms)':<15}")
    print("-"*80)
    
    for result in results:
        print(f"{result['name'][:50]:<50} {result['category']:<10} {result['total_direct_kernel_time_ms']:<15.2f}")
    
    # Export to CSV
    csv_file = "torch_prof/kernel_times_summary.csv"
    with open(csv_file, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['name', 'category', 'total_direct_kernel_time_ms', 'trace_file'])
        writer.writeheader()
        writer.writerows(results)
    
    print(f"\nResults exported to: {csv_file}")
    
    # Print top performers by category
    print("\n" + "="*80)
    print("TOP PERFORMERS BY CATEGORY")
    print("="*80)
    
    categories = {}
    for result in results:
        cat = result['category']
        if cat not in categories:
            categories[cat] = []
        categories[cat].append(result)
    
    for category, items in categories.items():
        print(f"\n{category} (Top 5):")
        for item in items[:5]:
            print(f"  {item['name']:<40} {item['total_direct_kernel_time_ms']:.2f} ms")
    
    # Summary statistics
    if results:
        total_time = sum(r['total_direct_kernel_time_ms'] for r in results)
        avg_time = total_time / len(results)
        max_time = max(r['total_direct_kernel_time_ms'] for r in results)
        min_time = min(r['total_direct_kernel_time_ms'] for r in results)
        
        print(f"\n" + "="*80)
        print("SUMMARY STATISTICS")
        print("="*80)
        print(f"Total profiles analyzed: {len(results)}")
        print(f"Total kernel time across all profiles: {total_time:.2f} ms")
        print(f"Average kernel time per profile: {avg_time:.2f} ms")
        print(f"Maximum kernel time: {max_time:.2f} ms")
        print(f"Minimum kernel time: {min_time:.2f} ms")


def extract_single_trace(trace_file):
    """Extract kernel time from a single trace file."""
    if not os.path.exists(trace_file):
        print(f"Error: File not found: {trace_file}")
        return
    
    kernel_time = extract_kernel_time_from_trace(trace_file)
    
    if kernel_time is not None:
        print(f"File: {trace_file}")
        print(f"Total Direct Kernel Time: {kernel_time:.2f} ms")
    else:
        print(f"Could not extract kernel time from: {trace_file}")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        # Extract from specific file
        extract_single_trace(sys.argv[1])
    else:
        # Analyze all files
        analyze_all_kernel_times()