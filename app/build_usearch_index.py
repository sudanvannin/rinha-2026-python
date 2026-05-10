from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from usearch.index import Index

from app.classifier import (
    DIMS,
    MAGIC,
    USEARCH_CONNECTIVITY,
    USEARCH_EXPANSION_ADD,
    USEARCH_EXPANSION_SEARCH,
)


def build_index(candidates_path: Path, output_path: Path) -> None:
    raw = candidates_path.read_bytes()
    if len(raw) < 16 or raw[:8] != MAGIC:
        raise RuntimeError(f"invalid candidate index: {candidates_path}")

    count = int.from_bytes(raw[8:12], "little")
    dims = int.from_bytes(raw[12:16], "little")
    if dims != DIMS:
        raise RuntimeError(f"invalid index dimensions: expected {DIMS}, got {dims}")

    vector_offset = 16
    vector_count = count * DIMS
    vectors = np.frombuffer(raw, dtype="<f4", count=vector_count, offset=vector_offset).reshape(count, DIMS)

    index = Index(
        ndim=DIMS,
        metric="l2sq",
        dtype="f32",
        connectivity=USEARCH_CONNECTIVITY,
        expansion_add=USEARCH_EXPANSION_ADD,
        expansion_search=USEARCH_EXPANSION_SEARCH,
    )
    index.add(np.arange(count, dtype=np.uint64), vectors)
    index.save(output_path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("candidates", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    build_index(args.candidates, args.output)


if __name__ == "__main__":
    main()
