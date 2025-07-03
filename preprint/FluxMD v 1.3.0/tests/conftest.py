"""Vendored FluxMD v1.3.0 test options."""

def pytest_addoption(parser):
    """Register ``--device`` **once** across the test session."""

    added = getattr(parser, "_addedargs", {})
    duplicate = any(
        getattr(opt, "dest", None) == "device"
        for opt in (added.values() if isinstance(added, dict) else added)
    )

    if duplicate:
        return

    try:
        parser.addoption(
            "--device",
            action="store",
            choices=["cpu", "cuda"],
            default="cpu",
            help="Device on which to run FluxMD tests",
        )
    except ValueError:
        # Another plugin raced us – ignore.
        pass