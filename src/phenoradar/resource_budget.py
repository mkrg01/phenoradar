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
                        available = int(limit) - current + int(stats.get("inactive_file", 0))
                        budget = min(budget, max(0, available))
                    except (OSError, ValueError):
                        continue

    except (OSError, ValueError):
        pass
    return budget
