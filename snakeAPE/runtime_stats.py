"""Process runtime statistics helpers."""

from __future__ import annotations

from pathlib import Path
import resource
import sys


def _linux_status_value_mb(field_name: str) -> float:
    """Return a Linux /proc/self/status KiB field converted to MiB."""

    status_path = Path("/proc/self/status")
    try:
        for line in status_path.read_text(encoding="utf-8").splitlines():
            if not line.startswith(f"{field_name}:"):
                continue
            parts = line.split()
            if len(parts) < 2:
                return 0.0
            value_kib = int(parts[1])
            if value_kib <= 0:
                return 0.0
            return value_kib / 1024.0
    except (FileNotFoundError, OSError, ValueError):
        return 0.0
    return 0.0


def current_peak_rss_mb() -> float:
    """Return the current process peak resident set size in MiB."""

    if sys.platform.startswith("linux"):
        peak_mb = _linux_status_value_mb("VmHWM")
        if peak_mb > 0:
            return peak_mb

    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if peak <= 0:
        return 0.0
    divisor = 1024 * 1024 if sys.platform == "darwin" else 1024
    return peak / divisor
