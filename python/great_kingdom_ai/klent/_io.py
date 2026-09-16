"""Atomic file copy helpers for the KLENT pipeline."""

from __future__ import annotations

from pathlib import Path

COPY_BUFFER_BYTES = 64 * 1024 * 1024


def copy_file_atomic(source: Path, destination: Path) -> None:
    """Copy ``source`` to ``destination`` so a partial write cannot be observed."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f"{destination.name}.tmp")
    try:
        _copy_file_buffered(source, temporary)
        temporary.replace(destination)
    finally:
        if temporary.exists():
            temporary.unlink()


def _copy_file_buffered(source: Path, destination: Path) -> None:
    buffer = bytearray(COPY_BUFFER_BYTES)
    view = memoryview(buffer)
    with source.open("rb", buffering=0) as source_file:
        with destination.open("wb", buffering=0) as destination_file:
            while True:
                read = source_file.readinto(buffer)
                if not read:
                    break
                destination_file.write(view[:read])


__all__ = ["COPY_BUFFER_BYTES", "copy_file_atomic"]
