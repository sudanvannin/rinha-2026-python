from __future__ import annotations

import unittest

import orjson

from app.classifier import classify_body, classify_fraud_count, init_classifier, normalize_request


LEGIT_PAYLOAD = {
    "id": "tx-smoke-001",
    "transaction": {
        "amount": 384.88,
        "installments": 3,
        "requested_at": "2026-03-11T20:23:35Z",
    },
    "customer": {
        "avg_amount": 769.76,
        "tx_count_24h": 3,
        "known_merchants": ["MERC-009", "MERC-001", "MERC-001"],
    },
    "merchant": {
        "id": "MERC-001",
        "mcc": "5912",
        "avg_amount": 298.95,
    },
    "terminal": {
        "is_online": False,
        "card_present": True,
        "km_from_home": 13.7090520965,
    },
    "last_transaction": {
        "timestamp": "2026-03-11T14:58:35Z",
        "km_from_current": 18.8626479774,
    },
}

FRAUD_PAYLOAD = {
    "id": "tx-fraud-001",
    "transaction": {
        "amount": 7500,
        "installments": 10,
        "requested_at": "2026-03-12T02:05:00Z",
    },
    "customer": {
        "avg_amount": 120,
        "tx_count_24h": 16,
        "known_merchants": ["MERC-001", "MERC-002"],
    },
    "merchant": {
        "id": "MERC-080",
        "mcc": "7995",
        "avg_amount": 60,
    },
    "terminal": {
        "is_online": True,
        "card_present": False,
        "km_from_home": 650,
    },
    "last_transaction": {
        "timestamp": "2026-03-12T02:00:00Z",
        "km_from_current": 720,
    },
}


class ClassifierTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        init_classifier()

    def test_normalization(self) -> None:
        vector = normalize_request(LEGIT_PAYLOAD)
        self.assertEqual(len(vector), 14)
        self.assertAlmostEqual(float(vector[0]), 0.0385, places=5)
        self.assertAlmostEqual(float(vector[1]), 0.25, places=5)
        self.assertAlmostEqual(float(vector[12]), 0.2, places=5)

    def test_approves_low_risk(self) -> None:
        self.assertLess(classify_fraud_count(LEGIT_PAYLOAD), 3)

    def test_denies_high_risk(self) -> None:
        self.assertGreaterEqual(classify_fraud_count(FRAUD_PAYLOAD), 3)

    def test_body_classifier_matches_object_classifier(self) -> None:
        body = orjson.dumps(FRAUD_PAYLOAD)
        self.assertEqual(classify_body(body), classify_fraud_count(FRAUD_PAYLOAD))


if __name__ == "__main__":
    unittest.main()
