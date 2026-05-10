from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import numpy as np
from usearch.index import Index

DIMS = 14
MAGIC = b"R26IDX1\n"
DEFAULT_INDEX_PATH = Path(__file__).resolve().parent.parent / "data" / "candidates.bin"
DEFAULT_USEARCH_PATH = Path(__file__).resolve().parent.parent / "data" / "candidates.usearch"
KNN_SCORE_MIN = float(os.getenv("RINHA_KNN_SCORE_MIN", "1.5"))
KNN_SCORE_MAX = float(os.getenv("RINHA_KNN_SCORE_MAX", "12.0"))
USEARCH_CONNECTIVITY = int(os.getenv("RINHA_USEARCH_CONNECTIVITY", "16"))
USEARCH_EXPANSION_ADD = int(os.getenv("RINHA_USEARCH_EXPANSION_ADD", "96"))
USEARCH_EXPANSION_SEARCH = int(os.getenv("RINHA_USEARCH_EXPANSION_SEARCH", "512"))

MCC_RISK = {
    "5411": 0.15,
    "5812": 0.30,
    "5912": 0.20,
    "5944": 0.45,
    "7801": 0.80,
    "7802": 0.75,
    "7995": 0.85,
    "4511": 0.35,
    "5311": 0.25,
    "5999": 0.50,
}

RESPONSES = (
    b'{"approved":true,"fraud_score":0}',
    b'{"approved":true,"fraud_score":0.2}',
    b'{"approved":true,"fraud_score":0.4}',
    b'{"approved":false,"fraud_score":0.6}',
    b'{"approved":false,"fraud_score":0.8}',
    b'{"approved":false,"fraud_score":1}',
)


@dataclass(slots=True)
class CandidateIndex:
    labels: np.ndarray
    index: Index


_INDEX: CandidateIndex | None = None


def _clamp01(value: float) -> float:
    if value < 0:
        return 0.0
    if value > 1:
        return 1.0
    return value


def _parse2(value: str, offset: int) -> int:
    return (ord(value[offset]) - 48) * 10 + ord(value[offset + 1]) - 48


def _parse4(value: str, offset: int) -> int:
    return (
        (ord(value[offset]) - 48) * 1000
        + (ord(value[offset + 1]) - 48) * 100
        + (ord(value[offset + 2]) - 48) * 10
        + ord(value[offset + 3])
        - 48
    )


def _timestamp_parts(ts: str) -> tuple[int, int, int, int, int, int]:
    return (
        _parse4(ts, 0),
        _parse2(ts, 5),
        _parse2(ts, 8),
        _parse2(ts, 11),
        _parse2(ts, 14),
        _parse2(ts, 17),
    )


def _days_from_civil(year: int, month: int, day: int) -> int:
    year -= month <= 2
    era = (year if year >= 0 else year - 399) // 400
    yoe = year - era * 400
    mp = month + (-3 if month > 2 else 9)
    doy = (153 * mp + 2) // 5 + day - 1
    doe = yoe * 365 + yoe // 4 - yoe // 100 + doy
    return era * 146097 + doe - 719468


def _epoch_seconds(ts: str) -> int:
    year, month, day, hour, minute, second = _timestamp_parts(ts)
    return ((_days_from_civil(year, month, day) * 24 + hour) * 60 + minute) * 60 + second


def _day_of_week_monday_zero(year: int, month: int, day: int) -> int:
    table = (0, 3, 2, 5, 0, 3, 5, 1, 4, 6, 2, 4)
    y = year - (1 if month < 3 else 0)
    dow = (y + y // 4 - y // 100 + y // 400 + table[month - 1] + day) % 7
    return (dow + 6) % 7


def normalize_request(request: dict[str, Any]) -> np.ndarray:
    tx = request["transaction"]
    customer = request["customer"]
    merchant = request["merchant"]
    terminal = request["terminal"]
    last = request.get("last_transaction")
    year, month, day, hour, _, _ = _timestamp_parts(tx["requested_at"])

    vector = np.empty(DIMS, dtype=np.float32)
    amount = float(tx["amount"])
    avg_amount = float(customer["avg_amount"])

    vector[0] = _clamp01(amount / 10000.0)
    vector[1] = _clamp01(float(tx["installments"]) / 12.0)
    vector[2] = _clamp01((amount / avg_amount) / 10.0) if avg_amount > 0 else 1.0
    vector[3] = hour / 23.0
    vector[4] = _day_of_week_monday_zero(year, month, day) / 6.0

    if last is not None:
        minutes = (_epoch_seconds(tx["requested_at"]) - _epoch_seconds(last["timestamp"])) / 60.0
        vector[5] = _clamp01(minutes / 1440.0)
        vector[6] = _clamp01(float(last["km_from_current"]) / 1000.0)
    else:
        vector[5] = -1.0
        vector[6] = -1.0

    vector[7] = _clamp01(float(terminal["km_from_home"]) / 1000.0)
    vector[8] = _clamp01(float(customer["tx_count_24h"]) / 20.0)
    vector[9] = 1.0 if terminal["is_online"] else 0.0
    vector[10] = 1.0 if terminal["card_present"] else 0.0
    vector[11] = 0.0 if merchant["id"] in customer["known_merchants"] else 1.0
    vector[12] = MCC_RISK.get(merchant["mcc"], 0.5)
    vector[13] = _clamp01(float(merchant["avg_amount"]) / 10000.0)
    vector[:] = np.floor(vector * 10000.0 + 0.5) / 10000.0
    return vector


def weighted_score(v: np.ndarray) -> float:
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


def load_index(path: Path = DEFAULT_INDEX_PATH, index_path: Path = DEFAULT_USEARCH_PATH) -> CandidateIndex:
    raw = path.read_bytes()
    if len(raw) < 16 or raw[:8] != MAGIC:
        raise RuntimeError(f"invalid candidate index: {path}")

    count = int.from_bytes(raw[8:12], "little")
    dims = int.from_bytes(raw[12:16], "little")
    if dims != DIMS:
        raise RuntimeError(f"invalid index dimensions: expected {DIMS}, got {dims}")

    vector_offset = 16
    vector_count = count * DIMS
    label_offset = vector_offset + vector_count * 4
    expected_len = label_offset + count
    if len(raw) != expected_len:
        raise RuntimeError(f"invalid index length: {path}")

    labels = np.frombuffer(raw, dtype=np.uint8, count=count, offset=label_offset).copy()

    if index_path.exists():
        index = Index.restore(index_path, view=True)
        if index is None:
            raise RuntimeError(f"invalid usearch index: {index_path}")
        index.expansion_search = USEARCH_EXPANSION_SEARCH
    else:
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

    return CandidateIndex(labels=labels, index=index)


def init_classifier(path: Path = DEFAULT_INDEX_PATH, index_path: Path = DEFAULT_USEARCH_PATH) -> CandidateIndex:
    global _INDEX
    _INDEX = load_index(path, index_path)
    return _INDEX


def _index() -> CandidateIndex:
    global _INDEX
    if _INDEX is None:
        _INDEX = load_index()
    return _INDEX


def _knn_fraud_count(vector: np.ndarray) -> int:
    index = _index()
    nearest = index.index.search(vector, 5).keys
    return int(index.labels[nearest].sum())


def classify_fraud_count(request: dict[str, Any]) -> int:
    vector = normalize_request(request)
    score = weighted_score(vector)
    if KNN_SCORE_MIN <= score < KNN_SCORE_MAX:
        fraud_count = _knn_fraud_count(vector)
        if fraud_count < 3 and _high_risk_late_boundary(vector, score):
            return 3
        return fraud_count
    return 5 if score >= 2.5 else 0


def classify_response(request: dict[str, Any]) -> bytes:
    return RESPONSES[classify_fraud_count(request)]


def _high_risk_late_boundary(vector: np.ndarray, score: float) -> bool:
    return (
        10.5 <= score < 11.0
        and vector[3] <= 7 / 23
        and vector[7] >= 0.35
        and vector[8] >= 0.40
        and vector[9] == 1
        and vector[10] == 0
        and vector[12] >= 0.80
    )
