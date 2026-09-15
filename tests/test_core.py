"""CPU smoke tests using synthetic features; no data or weights required."""

import sys
import unittest
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import eclt_features as core
import evaluate_pairs as pairs
import extract_features as extract


def feature_row(index):
    supported = index % 2
    layers = []
    for depth in (1, 2, 3):
        values = {key: (2 * supported - 1) * depth / 4 for key in core.RELATION_KEYS}
        values.update(layer=depth, predictor_norm=2.0, candidate_norm=3.0,
                      evidence_norm=1.0)
        layers.append(values)
    return {
        "claim_id": f"sample-{index}", "record_id": f"group-{index}",
        "labels": {"supported": supported},
        "features": {"evidence_span_present": True, "evidence_occurrence_count": 1,
                     "prompt_tokens": 10, "target_tokens": 2,
                     "confidence": {"mean_logprob": -1.0},
                     "layer_trajectory": layers, "trajectory_summary": {}},
    }


class CoreTests(unittest.TestCase):
    def test_pooling(self):
        hidden = torch.arange(24, dtype=torch.float32).reshape(1, 4, 6)
        self.assertTrue(torch.equal(extract.safe_mean(hidden, [1, 2]), hidden[0, 1:3].mean(0)))
        self.assertIsNone(extract.safe_mean(hidden, []))

    def test_occurrences(self):
        self.assertEqual(extract.exact_occurrences("ALPHA and alpha", "alpha"), [(0, 5), (10, 15)])

    def test_feature_contract(self):
        row = feature_row(1)
        full = core.features(row, "trajectory")
        relation = core.features(row, "relation_trajectory")
        self.assertIn("L1.predictor_norm", full)
        self.assertNotIn("L1.predictor_norm", relation)
        self.assertIn("L1.predictor_context_cos", relation)
        self.assertIn("confidence.mean_logprob", relation)

    def test_pure_features_exclude_controls(self):
        row = feature_row(0)
        values = pairs.pure_relation_features(row, last_only=False)
        self.assertTrue(values)
        self.assertFalse(any("norm" in key or "confidence" in key for key in values))
        self.assertNotIn("span_present", values)

    def test_grouped_probe(self):
        rows = [feature_row(i) for i in range(30)]
        labels = np.array([i % 2 for i in range(30)])
        groups = np.array([row["record_id"] for row in rows])
        scores = core.grouped_oof(rows, labels, groups, "relation_trajectory")
        self.assertTrue(np.isfinite(scores).all())
        self.assertEqual(core.score(labels, scores)["roc_auc"], 1.0)


if __name__ == "__main__":
    unittest.main()
