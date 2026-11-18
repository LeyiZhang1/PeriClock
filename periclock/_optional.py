"""Actionable errors for absent optional packages, preserving broken-import errors."""
from importlib import import_module


def require_dependency(module, extra):
    try:
        return import_module(module)
    except ModuleNotFoundError as exc:
        if exc.name != module:
            # A transitive dependency/import failure is not an absent integration.
            raise
        raise ImportError(
            f"Optional dependency '{module}' is required for this operation. "
            "From your PeriClock checkout, install the extra with: "
            f"python -m pip install -e '.[{extra}]' "
            f"(distribution extra: periclock[{extra}])."
        ) from exc
