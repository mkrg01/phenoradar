"""Ordered compressed worker audits, serialized to public TSV only by the parent."""
from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any

import polars as pl

AUDIT_BLOCK_ROWS = 65_536


class ColumnarAudits:
    def __init__(self, directory: Path) -> None:
        self.directory = directory
        self.frames: dict[str, list[pl.DataFrame]] = defaultdict(list)
        self.rows: dict[str, int] = defaultdict(int)
        self.parts: dict[str, int] = defaultdict(int)

    def __enter__(self) -> ColumnarAudits:
        return self

    def __exit__(self, *args: Any) -> None:
        for name in list(self.frames):
            self.flush(name)

    def append(self, name: str, frame: pl.DataFrame) -> None:
        self.frames[name].append(frame)
        self.rows[name] += frame.height
        if self.rows[name] >= AUDIT_BLOCK_ROWS:
            self.flush(name)

    def flush(self, name: str) -> None:
        if not self.frames[name]:
            return
        directory = self.directory / name
        directory.mkdir(parents=True, exist_ok=True)
        pl.concat(self.frames[name], how="vertical_relaxed").write_parquet(
            directory / f"{self.parts[name]:06d}.parquet",
        )
        self.parts[name] += 1
        self.frames[name].clear()
        self.rows[name] = 0


def merge_audits(source: Path, destination: Path) -> None:
    if not source.exists():
        return
    for directory in sorted(source.iterdir()):
        with (destination / directory.name).open("ab") as output:
            for part in sorted(directory.glob("*.parquet")):
                pl.read_parquet(part).write_csv(
                    output, separator="\t", include_header=output.tell() == 0,
                    line_terminator="\r\n",
                )
