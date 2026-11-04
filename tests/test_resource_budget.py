from pathlib import Path

import pytest

import phenoradar.resource_budget as resource


@pytest.mark.parametrize("inactive, expected", [(0, 1024), (1024, 2048)])
def test_memory_budget_honors_parent_job_limit(tmp_path: Path, monkeypatch,
                                             inactive: int, expected: int) -> None:
    proc = tmp_path / "proc"
    proc.mkdir()
    (proc / "meminfo").write_text("MemAvailable: 1048576 kB\n")
    (proc / "cgroup").write_text("0::/job/step\n")
    root = tmp_path / "cgroup"
    leaf = root / "job/step"
    leaf.mkdir(parents=True)
    (leaf / "memory.max").write_text("max")
    (root / "memory.max").write_text("max")
    job = root / "job"
    (job / "memory.max").write_text("4096")
    (job / "memory.current").write_text("3072")
    (job / "memory.stat").write_text(f"anon 1024\ninactive_file {inactive}\n")
    paths = {"/proc/meminfo": proc / "meminfo", "/proc/self/cgroup": proc / "cgroup",
             "/sys/fs/cgroup": root}
    monkeypatch.setattr(resource, "Path", lambda name: paths[name])
    assert resource.memory_budget() == expected
