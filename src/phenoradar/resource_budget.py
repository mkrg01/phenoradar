"""Available memory within the current process allocation."""
from __future__ import annotations

import os
from pathlib import Path


def memory_budget() -> int:
    """Use the allocation limit, counting reclaimable inactive file cache as available."""
    try:
        budget = int(os.sysconf("SC_AVPHYS_PAGES") * os.sysconf("SC_PAGE_SIZE"))
    except (OSError, ValueError):
        budget = 512 * 1024**2
    try:
        lines = Path("/proc/meminfo").read_text().splitlines()
        meminfo = dict(line.split(":", 1) for line in lines)
        budget = int(meminfo["MemAvailable"].split()[0]) * 1024
    except (OSError, ValueError, KeyError):
        pass
    # Grid Engine can enforce the job's virtual-memory allocation with a
    # process resource limit even when the memory cgroup has no job limit.
    try:
        from resource import RLIM_INFINITY, RLIMIT_AS, getrlimit

        address_space, _ = getrlimit(RLIMIT_AS)
        if address_space != RLIM_INFINITY:
            budget = min(budget, address_space)
    except (ImportError, OSError, ValueError):
        pass
    try:
        for line in Path("/proc/self/cgroup").read_text().splitlines():
            if line.startswith("0::"):
                root = Path("/sys/fs/cgroup")
                path = root / line[3:].lstrip("/")
                for scope in [path, *path.parents]:
                    if not scope.is_relative_to(root):
                        break
                    try:
                        limit = (scope / "memory.max").read_text().strip()
                        if limit == "max":
                            continue
                        lines = (scope / "memory.stat").read_text().splitlines()
                        stats = dict(row.split() for row in lines)
                        current = int((scope / "memory.current").read_text())
                        inactive = max(0, int(stats.get("inactive_file", 0)))
                        # Kernel counters can be sampled at different instants.
                        # Reclaimable pages cannot exceed total file/current pages.
                        reclaimable = min(inactive, current, int(stats.get("file", inactive)))
                        available = int(limit) - current + max(0, reclaimable)
                        budget = min(budget, max(0, available))
                    except (OSError, ValueError):
                        continue

    except (OSError, ValueError):
        pass
    return budget
