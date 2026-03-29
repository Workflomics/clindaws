"""Process runtime statistics helpers."""

from __future__ import annotations

import resource
import sys


def current_peak_rss_mb() -> float:
    """Return the current process peak resident set size in MiB."""

    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if peak <= 0:
        return 0.0
    divisor = 1024 * 1024 if sys.platform == "darwin" else 1024
    return peak / divisor
