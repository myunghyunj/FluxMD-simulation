"""CPU-related utilities."""
import multiprocessing as mp
from typing import Optional, Union

def parse_workers(value: Optional[Union[str, int]]) -> int:
    """Parse worker count from user input with auto-detection.

    - "auto", None, or "" -> cpu_count - 1 (min 1)
    - integer or string integer -> the given number (min 1)
    
    Raises:
        ValueError: if value is not a valid integer representation.
    """
    if value in (None, "", "auto"):
        # Auto-detect: use all cores minus 1, but at least 1
        return max(1, mp.cpu_count() - 1)
    
    try:
        workers = int(value)
    except (ValueError, TypeError) as e:
        raise ValueError(f"Invalid worker count: '{value}'. Must be an integer or 'auto'.") from e
    
    if workers < 1:
        raise ValueError(f"Worker count must be >= 1, got {workers}")
        
    return workers

def get_optimal_workers() -> int:
    """Get optimal number of workers for the current system.
    
    Returns:
        Optimal worker count (cpu_count - 1, minimum 1)
    """
    return max(1, mp.cpu_count() - 1)


def format_workers_info(n_workers: int) -> str:
    """Format worker information for display.
    
    Args:
        n_workers: Number of workers
        
    Returns:
        Formatted string describing worker configuration
    """
    total_cores = mp.cpu_count()
    if n_workers == 1:
        return f"1 worker (serial processing, {total_cores} cores available)"
    elif n_workers == max(1, total_cores - 1):
        return f"{n_workers} workers (auto-detected, {total_cores} cores total)"
    else:
        return f"{n_workers} workers ({total_cores} cores total)" 