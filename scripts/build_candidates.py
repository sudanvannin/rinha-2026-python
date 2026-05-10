#!/usr/bin/env python3
import argparse
import gzip
import json
import struct
from pathlib import Path

DIMS = 14
MAGIC = b"R26IDX1\n"


def weighted_score(v):
    score = 0.0

    if v[2] >= 0.20:
        score += 2.0
    elif v[2] >= 0.12:
        score += 1.0

    if v[12] >= 0.75:
        score += 2.0
    elif v[12] >= 0.45:
        score += 0.6

    if v[7] >= 0.20:
        score += 1.5
    elif v[7] >= 0.08:
        score += 0.8

    if v[8] >= 0.40:
        score += 1.2
    elif v[8] >= 0.25:
        score += 0.5

    if v[1] >= 0.50:
        score += 1.0
    elif v[1] >= 0.33:
        score += 0.4

    if v[11] >= 0.5:
        score += 1.0

    if v[0] >= 0.20:
        score += 1.0
    elif v[0] >= 0.08:
        score += 0.4

    if v[6] >= 0.20:
        score += 1.0
    elif v[6] >= 0.05:
        score += 0.4

    if 0 <= v[5] <= 10 / 1440:
        score += 0.8
    elif 0 <= v[5] <= 120 / 1440:
        score += 0.3

    if v[3] <= 6 / 23:
        score += 0.8
    if v[9] == 1 and v[10] == 0:
        score += 0.3
    if v[13] <= 0.012:
        score += 0.2

    return score


def load_references(path):
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as fp:
        return json.load(fp)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--references", required=True, type=Path)
    parser.add_argument("--output", default=Path("data/candidates.bin"), type=Path)
    parser.add_argument("--min-score", default=1.0, type=float)
    parser.add_argument("--max-score", default=12.0, type=float)
    args = parser.parse_args()

    refs = load_references(args.references)
    selected = []
    labels = bytearray()

    for item in refs:
        vector = item["vector"]
        score = weighted_score(vector)
        if args.min_score <= score < args.max_score:
            selected.append(vector)
            labels.append(1 if item["label"] == "fraud" else 0)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("wb") as fp:
        fp.write(MAGIC)
        fp.write(struct.pack("<II", len(selected), DIMS))
        for vector in selected:
            fp.write(struct.pack("<14f", *vector))
        fp.write(labels)

    print(f"wrote {args.output} with {len(selected)} candidates")


if __name__ == "__main__":
    main()
