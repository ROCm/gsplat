import torch
import os
import re
from contextlib import contextmanager
from typing import Optional, List, Dict, Any
import json
from datetime import datetime


class TorchProfiler:
    """Centralized torch profiler for gsplat kernels and operations."""
    
    def __init__(self, 
                 log_dir: str = "./torch_prof/logs",
                 enable_cuda: bool = True,
                 enable_stack: bool = True,
                 enable_memory: bool = True,
                 row_limit: int = 20):
        self.log_dir = log_dir
        self.enable_cuda = enable_cuda and torch.cuda.is_available()
        self.enable_stack = enable_stack
        self.enable_memory = enable_memory
        self.row_limit = row_limit
        
        # Create log directory if it doesn't exist
        os.makedirs(self.log_dir, exist_ok=True)
        
        # Define profiler activities
        self.activities = [torch.profiler.ProfilerActivity.CPU]
        if self.enable_cuda:
            self.activities.append(torch.profiler.ProfilerActivity.CUDA)
    
    def sanitize_name(self, name: str) -> str:
        """Sanitize names for directory/file naming."""
        return re.sub(r'[:\[\]\-/\s]', '_', name)
    
    @contextmanager
    def profile(self, name: str, group: Optional[str] = None):
        """Context manager for profiling a specific operation."""
        sanitized_name = self.sanitize_name(name)
        if group:
            sanitized_group = self.sanitize_name(group)
            trace_dir = os.path.join(self.log_dir, sanitized_group, sanitized_name)
        else:
            trace_dir = os.path.join(self.log_dir, sanitized_name)
        
        print(f"\n[Profiler] Starting profile for: {name}")
        print(f"[Profiler] Trace will be saved to: {trace_dir}")
        
        with torch.profiler.profile(
            activities=self.activities,
            on_trace_ready=torch.profiler.tensorboard_trace_handler(trace_dir),
            record_shapes=True,
            profile_memory=self.enable_memory,
            with_stack=self.enable_stack
        ) as prof:
            yield prof
        
        # Print summary
        print(f"\n[Profiler] Results for {name}:")
        # Check if CUDA timing is actually available in the profiler output
        key_averages = prof.key_averages()
        if key_averages and hasattr(key_averages[0], 'cuda_time_total') and self.enable_cuda:
            sort_key = "cuda_time_total"
        else:
            sort_key = "cpu_time_total"
        print(prof.key_averages().table(sort_by=sort_key, row_limit=self.row_limit))
        
        # Save additional metadata
        self._save_metadata(trace_dir, name, prof)
    
    def _save_metadata(self, trace_dir: str, name: str, prof):
        """Save profiling metadata to JSON."""
        os.makedirs(trace_dir, exist_ok=True)
        metadata = {
            "name": name,
            "timestamp": datetime.now().isoformat(),
            "cuda_enabled": self.enable_cuda,
            "memory_profiling": self.enable_memory,
            "stack_enabled": self.enable_stack,
        }
        
        # Extract key metrics
        key_averages = prof.key_averages()
        top_ops = []
        
        # Sort by appropriate time metric
        if self.enable_cuda and hasattr(key_averages[0] if key_averages else None, 'cuda_time_total'):
            sort_key = lambda x: x.cuda_time_total
        else:
            sort_key = lambda x: x.cpu_time_total
            
        for evt in sorted(key_averages, key=sort_key, reverse=True)[:10]:
            op_info = {
                "name": evt.key,
                "cpu_time_total": evt.cpu_time_total,
                "count": evt.count,
            }
            
            # Add CUDA time if available
            if hasattr(evt, 'cuda_time_total'):
                op_info["cuda_time_total"] = evt.cuda_time_total
            else:
                op_info["cuda_time_total"] = 0
                
            # Add memory usage if available
            if self.enable_memory and hasattr(evt, 'cuda_memory_usage'):
                op_info["cuda_memory_usage"] = evt.cuda_memory_usage
                
            top_ops.append(op_info)
        
        metadata["top_operations"] = top_ops
        
        with open(os.path.join(trace_dir, "metadata.json"), "w") as f:
            json.dump(metadata, f, indent=2)


# Global profiler instance
_profiler = None

def get_profiler(**kwargs) -> TorchProfiler:
    """Get or create the global profiler instance."""
    global _profiler
    if _profiler is None:
        _profiler = TorchProfiler(**kwargs)
    return _profiler


@contextmanager
def profile_code(name: str, group: Optional[str] = None, **profiler_kwargs):
    """Convenient function to profile code blocks."""
    profiler = get_profiler(**profiler_kwargs)
    with profiler.profile(name, group) as prof:
        yield prof