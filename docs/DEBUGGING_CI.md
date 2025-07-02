# Debugging CI

To enable verbose output in GitHub Actions, set the environment variable `ACTIONS_STEP_DEBUG` to `true` for a job or a single step:

```yaml
env:
  ACTIONS_STEP_DEBUG: true
```

Combine with `set -x` inside shell steps to echo each command.
