from pathlib import Path

import pytest

import phenoradar.resource_budget as resource


@pytest.mark.parametrize("inactive, file_bytes, expected", [
    (0, None, 1024), (1024, None, 2048), (10000, 512, 1536), (10000, 10000, 4096),
])
def test_memory_budget_honors_parent_job_limit(tmp_path: Path, monkeypatch,
                                             inactive: int, file_bytes: int | None,
                                             expected: int) -> None:
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
    stat = f"anon 1024\ninactive_file {inactive}\n"
    if file_bytes is not None:
        stat += f"file {file_bytes}\n"
    (job / "memory.stat").write_text(stat)
    paths = {"/proc/meminfo": proc / "meminfo", "/proc/self/cgroup": proc / "cgroup",
             "/sys/fs/cgroup": root}
    monkeypatch.setattr(resource, "Path", lambda name: paths[name])
    assert resource.memory_budget() == expected


@pytest.mark.parametrize("address_space, expected", [(4096, 4096), (-1, 1048576)])
def test_memory_budget_honors_grid_engine_address_space_limit(
    tmp_path: Path, monkeypatch, address_space: int, expected: int,
) -> None:
    posix_resource = pytest.importorskip("resource")
    meminfo = tmp_path / "meminfo"
    meminfo.write_text("MemAvailable: 1024 kB\n")
    cgroup = tmp_path / "cgroup"
    cgroup.write_text("3:cpuset:/AGE/123.1\n")
    paths = {"/proc/meminfo": meminfo, "/proc/self/cgroup": cgroup}
    limit = posix_resource.RLIM_INFINITY if address_space < 0 else address_space
    monkeypatch.setattr(resource, "Path", lambda name: paths[name])
    monkeypatch.setattr(posix_resource, "getrlimit", lambda _: (limit, limit))
    assert resource.memory_budget() == expected
