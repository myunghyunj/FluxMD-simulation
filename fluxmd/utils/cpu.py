"""CPU-related utilities."""
import multiprocessing as mp
from typing import Optional, Union


def parse_workers(value: Optional[Union[str, int]]) -> int:
    """Parse worker count from user input with auto-detection.

    - ``"auto"``, ``None`` or ``""`` -> ``cpu_count - 1`` (min 1)
    - integer or string integer -> clamped to a minimum of 1

    Raises
    ------
    ValueError
        If ``value`` cannot be converted to an integer and is not ``"auto"``.
    """

    if value in (None, "", "auto"):
        # Auto-detect: use all cores minus 1, but at least 1
        return max(1, mp.cpu_count() - 1)

    try:
        workers = int(value)
    except (ValueError, TypeError) as e:
        raise ValueError(
            f"Invalid worker count: '{value}'. Must be 'auto' or a positive integer."
        ) from e

    # Clamp to a minimum of 1 rather than raising an error to be more forgiving
    return max(1, workers)


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
