"""Process runtime statistics helpers."""

from __future__ import annotations

from bisect import bisect_right
import os
from pathlib import Path
import resource
import subprocess
import sys
import threading
from time import perf_counter_ns


def _status_value_kib(status_path: Path, field_name: str) -> int:
    """Return one KiB-valued field from a `/proc/<pid>/status` file."""

    try:
        for line in status_path.read_text(encoding="utf-8").splitlines():
            if not line.startswith(f"{field_name}:"):
                continue
            parts = line.split()
            if len(parts) < 2:
                return 0
            value_kib = int(parts[1])
            return max(value_kib, 0)
    except (FileNotFoundError, OSError, ValueError):
        return 0
    return 0


def _linux_status_value_mb(field_name: str) -> float:
    """Return a Linux /proc/self/status KiB field converted to MiB."""

    value_kib = _status_value_kib(Path("/proc/self/status"), field_name)
    if value_kib <= 0:
        return 0.0
    return value_kib / 1024.0


def current_rss_mb() -> float:
    """Return the best-effort current process resident set size in MiB."""

    if sys.platform.startswith("linux"):
        current_mb = _linux_status_value_mb("VmRSS")
        if current_mb > 0:
            return current_mb

    try:
        output = subprocess.run(
            ["ps", "-o", "rss=", "-p", str(os.getpid())],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return current_peak_rss_mb()

    try:
        value_kib = int(output)
    except ValueError:
        return current_peak_rss_mb()
    if value_kib <= 0:
        return current_peak_rss_mb()
    return value_kib / 1024.0


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


def _linux_process_tree_rss_mb(root_pid: int) -> float:
    """Return the current RSS of one process tree on Linux in MiB."""

    proc_root = Path("/proc")
    parent_by_pid: dict[int, int] = {}
    rss_by_pid: dict[int, int] = {}

    try:
        proc_entries = tuple(proc_root.iterdir())
    except OSError:
        return 0.0

    for entry in proc_entries:
        if not entry.name.isdigit():
            continue
        pid = int(entry.name)
        status_path = entry / "status"
        try:
            ppid = 0
            rss_kib = 0
            for line in status_path.read_text(encoding="utf-8").splitlines():
                if line.startswith("PPid:"):
                    parts = line.split()
                    if len(parts) >= 2:
                        ppid = int(parts[1])
                elif line.startswith("VmRSS:"):
                    parts = line.split()
                    if len(parts) >= 2:
                        rss_kib = int(parts[1])
                if ppid and rss_kib:
                    break
            parent_by_pid[pid] = ppid
            rss_by_pid[pid] = max(rss_kib, 0)
        except (FileNotFoundError, OSError, ValueError):
            continue

    children_by_pid: dict[int, list[int]] = {}
    for pid, ppid in parent_by_pid.items():
        children_by_pid.setdefault(ppid, []).append(pid)

    total_kib = 0
    pending = [root_pid]
    seen: set[int] = set()
    while pending:
        pid = pending.pop()
        if pid in seen:
            continue
        seen.add(pid)
        total_kib += rss_by_pid.get(pid, 0)
        pending.extend(children_by_pid.get(pid, ()))

    return total_kib / 1024.0 if total_kib > 0 else 0.0


def _ps_process_tree_rss_mb(root_pid: int) -> float:
    """Return the current RSS of one process tree using `ps` output."""

    try:
        output = subprocess.run(
            ["ps", "-axo", "pid=,ppid=,rss="],
            check=True,
            capture_output=True,
            text=True,
        ).stdout
    except (OSError, subprocess.SubprocessError):
        return 0.0

    parent_by_pid: dict[int, int] = {}
    rss_by_pid: dict[int, int] = {}
    for line in output.splitlines():
        parts = line.split()
        if len(parts) != 3:
            continue
        try:
            pid, ppid, rss_kib = (int(part) for part in parts)
        except ValueError:
            continue
        parent_by_pid[pid] = ppid
        rss_by_pid[pid] = max(rss_kib, 0)

    children_by_pid: dict[int, list[int]] = {}
    for pid, ppid in parent_by_pid.items():
        children_by_pid.setdefault(ppid, []).append(pid)

    total_kib = 0
    pending = [root_pid]
    seen: set[int] = set()
    while pending:
        pid = pending.pop()
        if pid in seen:
            continue
        seen.add(pid)
        total_kib += rss_by_pid.get(pid, 0)
        pending.extend(children_by_pid.get(pid, ()))

    return total_kib / 1024.0 if total_kib > 0 else 0.0


def sample_process_tree_rss_mb(root_pid: int | None = None) -> float:
    """Return the best-effort current combined RSS for one process tree in MiB."""

    effective_root_pid = root_pid if root_pid is not None else os.getpid()
    if sys.platform.startswith("linux"):
        rss_mb = _linux_process_tree_rss_mb(effective_root_pid)
        if rss_mb > 0:
            return rss_mb

    rss_mb = _ps_process_tree_rss_mb(effective_root_pid)
    if rss_mb > 0:
        return rss_mb

    if effective_root_pid == os.getpid():
        return current_rss_mb()
    return 0.0


class ProcessTreePeakMonitor:
    """Sample and retain the peak RSS of one process tree over time."""

    def __init__(
        self,
        *,
        root_pid: int | None = None,
        sample_interval_sec: float = 0.1,
    ) -> None:
        self.root_pid = root_pid if root_pid is not None else os.getpid()
        self.sample_interval_sec = max(sample_interval_sec, 0.01)
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._peak_mb = 0.0
        self._peak_timestamps_ns: list[int] = []
        self._peak_values_mb: list[float] = []

    def __enter__(self) -> "ProcessTreePeakMonitor":
        self.start()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.stop()

    def start(self) -> None:
        if self._thread is not None:
            return
        self.sample_now()
        self._thread = threading.Thread(
            target=self._run,
            name="clindaws-process-tree-peak-monitor",
            daemon=True,
        )
        self._thread.start()

    def stop(self) -> None:
        self.sample_now()
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=self.sample_interval_sec * 2)
            self._thread = None
        self.sample_now()

    def sample_now(self) -> float:
        current_mb = sample_process_tree_rss_mb(self.root_pid)
        timestamp_ns = perf_counter_ns()
        with self._lock:
            if current_mb > self._peak_mb:
                self._peak_mb = current_mb
                self._peak_timestamps_ns.append(timestamp_ns)
                self._peak_values_mb.append(current_mb)
            elif not self._peak_timestamps_ns:
                self._peak_timestamps_ns.append(timestamp_ns)
                self._peak_values_mb.append(self._peak_mb)
        return current_mb

    def current_peak_mb(self) -> float:
        with self._lock:
            return self._peak_mb

    def peak_at(self, timestamp_ns: int) -> float:
        with self._lock:
            if not self._peak_timestamps_ns:
                return 0.0
            index = bisect_right(self._peak_timestamps_ns, timestamp_ns) - 1
            if index < 0:
                return 0.0
            return self._peak_values_mb[index]

    def _run(self) -> None:
        while not self._stop_event.wait(self.sample_interval_sec):
            self.sample_now()
