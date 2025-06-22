# Fix for n_workers TypeError Issue

## Problem
When users press Enter at the "Number of parallel workers" prompt (accepting the default), the MatryoshkaTrajectoryGenerator was getting `None` for `n_workers`, causing a TypeError when compared to an integer:

```
TypeError: '>' not supported between instances of 'NoneType' and 'int'
```

## Root Cause
The `n_workers` parameter was not being properly handled when it was `None`, despite the `parse_workers()` function being designed to handle this case.

## Solution Applied

### 1. Defensive Initialization (matryoshka_generator.py)
Added defensive handling in the `__init__` method:
```python
# Parse workers with defensive handling
n_workers_param = params.get('n_workers')
try:
    self.n_workers = parse_workers(n_workers_param)
except Exception as e:
    print(f"Warning: Failed to parse n_workers={n_workers_param}, using auto-detection. Error: {e}")
    self.n_workers = parse_workers(None)

# Ensure n_workers is never None
if self.n_workers is None:
    self.n_workers = parse_workers(None)
```

### 2. Runtime Safety Check (matryoshka_generator.py)
Added a safety check in the `run()` method:
```python
# Ensure n_workers is valid
if self.n_workers is None:
    self.n_workers = parse_workers(None)  # Will return auto-detected value
```

### 3. Comprehensive Tests (test_n_workers_fix.py)
Added unit tests to verify the fix handles all edge cases:
- `n_workers = None`
- `n_workers = ''` (empty string)
- `n_workers = 'auto'`
- `n_workers = 4` (integer)
- `n_workers = '3'` (string integer)
- Missing n_workers parameter
- Invalid n_workers values

## Result
The fix ensures that `n_workers` is always a valid integer (≥1) regardless of user input, preventing the TypeError and allowing the Matryoshka workflow to run successfully.