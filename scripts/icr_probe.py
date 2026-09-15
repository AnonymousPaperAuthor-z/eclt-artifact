#!/usr/bin/env python3
"""Evaluate the official-code-aligned ICR Probe on the frozen SciFact split."""

from __future__ import annotations

import argparse
import copy
import json
import random
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, balanced_accuracy_score, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset


SEED = 20260807


def iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                row = json.loads(line)
                if isinstance(row, dict):
                    yield row


def labels(rows: list[dict[str, Any]]) -> np.ndarray:
    return np.asarray([int((row.get("labels") or {}).get("supported", 0)) for row in rows])


def matrix(rows: list[dict[str, Any]], key: str) -> np.ndarray:
    values = []
    for row in rows:
        feature = row.get("features") or {}
        if key == "icr":
            vector = feature.get("icr_score_by_layer") or []
        elif key == "confidence":
            confidence = feature.get("confidence") or {}
            vector = [
                confidence.get("avg_logprob"),
                confidence.get("min_logprob"),
                confidence.get("avg_entropy"),
                confidence.get("avg_top2_margin"),
            ]
        else:
            raise ValueError(key)
        values.append([float(value) if value is not None else np.nan for value in vector])
    result = np.asarray(values, dtype=np.float64)
    if result.ndim != 2 or not result.shape[1]:
        raise ValueError(f"empty feature matrix for {key}: {result.shape}")
    return result


def linear_model() -> Pipeline:
    return Pipeline(
        [
            ("imputer", SimpleImputer(strategy="constant", fill_value=0.0)),
            ("scaler", StandardScaler()),
            (
                "classifier",
                LogisticRegression(
                    C=0.1,
                    class_weight="balanced",
                    max_iter=3000,
                    solver="liblinear",
                    random_state=SEED,
                ),
            ),
        ]
    )


class ICRProbe(nn.Module):
    """Official ICR Probe architecture: L -> 128 -> 64 -> 32 -> 1."""

    def __init__(self, input_dim: int) -> None:
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.bn1 = nn.BatchNorm1d(128)
        self.fc2 = nn.Linear(128, 64)
        self.bn2 = nn.BatchNorm1d(64)
        self.fc3 = nn.Linear(64, 32)
        self.bn3 = nn.BatchNorm1d(32)
        self.fc4 = nn.Linear(32, 1)
        self.dropout = nn.Dropout(0.3)
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.kaiming_uniform_(module.weight, a=0.01, nonlinearity="leaky_relu")
                nn.init.zeros_(module.bias)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        values = self.dropout(F.leaky_relu(self.bn1(self.fc1(values)), 0.01))
        values = self.dropout(F.leaky_relu(self.bn2(self.fc2(values)), 0.01))
        values = self.dropout(F.leaky_relu(self.bn3(self.fc3(values)), 0.01))
        return torch.sigmoid(self.fc4(values)).squeeze(-1)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)


def fit_icr_probe(train_x: np.ndarray, train_y: np.ndarray, test_x: np.ndarray) -> np.ndarray:
    seed_everything(SEED)
    imputer = SimpleImputer(strategy="constant", fill_value=0.0)
    scaler = StandardScaler()
    train_x = scaler.fit_transform(imputer.fit_transform(train_x)).astype(np.float32)
    test_x = scaler.transform(imputer.transform(test_x)).astype(np.float32)
    dataset = TensorDataset(torch.from_numpy(train_x), torch.from_numpy(train_y.astype(np.float32)))
    generator = torch.Generator().manual_seed(SEED)
    loader = DataLoader(dataset, batch_size=16, shuffle=True, generator=generator, drop_last=False)
    model = ICRProbe(train_x.shape[1])
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    model.train()
    for _ in range(100):
        for batch_x, batch_y in loader:
            if len(batch_x) == 1:
                continue
            optimizer.zero_grad()
            probability = model(batch_x)
            loss = F.binary_cross_entropy(probability, batch_y)
            loss.backward()
            optimizer.step()
    model.eval()
    with torch.no_grad():
        return model(torch.from_numpy(test_x)).numpy().astype(np.float64)


def score(y: np.ndarray, probability: np.ndarray) -> dict[str, Any]:
    return {
        "n": int(len(y)),
        "positives": int(y.sum()),
        "roc_auc": float(roc_auc_score(y, probability)),
        "average_precision": float(average_precision_score(y, probability)),
        "balanced_accuracy": float(balanced_accuracy_score(y, probability >= 0.5)),
    }


def grouped_oof(
    values: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    method: str,
) -> np.ndarray:
    splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED)
    probability = np.full(len(y), np.nan)
    for train, test in splitter.split(values, y, groups):
        if method == "mlp":
            probability[test] = fit_icr_probe(values[train], y[train], values[test])
        else:
            model = linear_model()
            model.fit(values[train], y[train])
            probability[test] = model.predict_proba(values[test])[:, 1]
    if not np.isfinite(probability).all():
        raise AssertionError(f"incomplete OOF probabilities for {method}")
    return probability


def transfer(
    train_x: np.ndarray,
    train_y: np.ndarray,
    test_x: np.ndarray,
    method: str,
) -> np.ndarray:
    if method == "mlp":
        return fit_icr_probe(train_x, train_y, test_x)
    model = linear_model()
    model.fit(train_x, train_y)
    return model.predict_proba(test_x)[:, 1]


def grouped_bootstrap_delta(
    y: np.ndarray,
    baseline: np.ndarray,
    proposed: np.ndarray,
    groups: np.ndarray,
    iterations: int,
) -> dict[str, Any]:
    unique = np.unique(groups)
    group_indices = {group: np.flatnonzero(groups == group) for group in unique}
    rng = np.random.default_rng(SEED)
    deltas = []
    for _ in range(iterations):
        sampled = rng.choice(unique, size=len(unique), replace=True)
        selected = np.concatenate([group_indices[group] for group in sampled])
        if len(np.unique(y[selected])) < 2:
            continue
        deltas.append(
            roc_auc_score(y[selected], proposed[selected])
            - roc_auc_score(y[selected], baseline[selected])
        )
    values = np.asarray(deltas)
    return {
        "iterations": int(len(values)),
        "mean": float(values.mean()),
        "ci95_low": float(np.quantile(values, 0.025)),
        "ci95_high": float(np.quantile(values, 0.975)),
    }


def evaluate_slice(rows: list[dict[str, Any]], bootstrap: int) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    y = labels(rows)
    groups = np.asarray([str(row.get("record_id")) for row in rows])
    icr = matrix(rows, "icr")
    confidence = matrix(rows, "confidence")
    predictions = {
        "confidence": grouped_oof(confidence, y, groups, "linear"),
        "icr_linear": grouped_oof(icr, y, groups, "linear"),
        "icr_probe": grouped_oof(icr, y, groups, "mlp"),
    }
    metrics = {name: score(y, value) for name, value in predictions.items()}
    metrics["deltas"] = {
        "icr_probe_minus_confidence": grouped_bootstrap_delta(
            y, predictions["confidence"], predictions["icr_probe"], groups, bootstrap
        ),
        "icr_probe_minus_icr_linear": grouped_bootstrap_delta(
            y, predictions["icr_linear"], predictions["icr_probe"], groups, bootstrap
        ),
    }
    return metrics, predictions


def render(report: dict[str, Any]) -> str:
    lines = [
        "# ICR Probe SciFact Baseline",
        "",
        f"- rows: {report['rows']}",
        f"- feature layers: {report['feature_layers']}",
        f"- upstream commit: `{report['upstream_commit']}`",
        "- classifier: official L-128-64-32-1 MLP, 100 fixed epochs",
        "- protocol: identical SciFact labels and document groups as ECLT",
        "",
    ]
    for title, key in (
        ("All 773, document-grouped OOF", "all_document_grouped_oof"),
        ("Train to 85 record-disjoint dev", "dev_record_disjoint_transfer"),
    ):
        lines.extend(
            [
                f"## {title}",
                "",
                "| Method | ROC AUC | AP | Balanced accuracy |",
                "|---|---:|---:|---:|",
            ]
        )
        for method in ("confidence", "icr_linear", "icr_probe"):
            metric = report[key][method]
            lines.append(
                f"| {method} | {metric['roc_auc']:.4f} | {metric['average_precision']:.4f} | "
                f"{metric['balanced_accuracy']:.4f} |"
            )
        lines.extend(["", f"Grouped bootstrap: `{report[key]['deltas']}`", ""])
    lines.extend(["## Frozen interpretation", "", f"- {report['interpretation']}", ""])
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-md", type=Path, required=True)
    parser.add_argument("--bootstrap", type=int, default=2000)
    args = parser.parse_args()

    rows = [row for row in iter_jsonl(args.features) if row.get("status") == "ok"]
    if len(rows) != 773:
        raise ValueError(f"expected 773 valid rows, got {len(rows)}")
    ids = [str(row.get("claim_id")) for row in rows]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate claim IDs")
    layer_counts = {len((row.get("features") or {}).get("icr_score_by_layer") or []) for row in rows}
    if len(layer_counts) != 1 or not next(iter(layer_counts)):
        raise ValueError(f"inconsistent ICR layer counts: {layer_counts}")

    train = [row for row in rows if row.get("partition") == "train"]
    dev = [row for row in rows if row.get("partition") == "dev"]
    train_records = {str(row.get("record_id")) for row in train}
    dev_disjoint = [row for row in dev if str(row.get("record_id")) not in train_records]
    if len(train) != 564 or len(dev_disjoint) != 85:
        raise ValueError(f"unexpected partitions train={len(train)} strict_dev={len(dev_disjoint)}")

    all_metrics, _ = evaluate_slice(rows, args.bootstrap)
    train_y = labels(train)
    dev_y = labels(dev_disjoint)
    train_icr = matrix(train, "icr")
    dev_icr = matrix(dev_disjoint, "icr")
    train_confidence = matrix(train, "confidence")
    dev_confidence = matrix(dev_disjoint, "confidence")
    strict_predictions = {
        "confidence": transfer(train_confidence, train_y, dev_confidence, "linear"),
        "icr_linear": transfer(train_icr, train_y, dev_icr, "linear"),
        "icr_probe": transfer(train_icr, train_y, dev_icr, "mlp"),
    }
    strict_metrics = {name: score(dev_y, value) for name, value in strict_predictions.items()}
    strict_groups = np.asarray([str(row.get("record_id")) for row in dev_disjoint])
    strict_metrics["deltas"] = {
        "icr_probe_minus_confidence": grouped_bootstrap_delta(
            dev_y,
            strict_predictions["confidence"],
            strict_predictions["icr_probe"],
            strict_groups,
            args.bootstrap,
        ),
        "icr_probe_minus_icr_linear": grouped_bootstrap_delta(
            dev_y,
            strict_predictions["icr_linear"],
            strict_predictions["icr_probe"],
            strict_groups,
            args.bootstrap,
        ),
    }

    pooled_gain = all_metrics["icr_probe"]["roc_auc"] - all_metrics["confidence"]["roc_auc"]
    strict_gain = strict_metrics["icr_probe"]["roc_auc"] - strict_metrics["confidence"]["roc_auc"]
    if pooled_gain > 0 and strict_gain > 0:
        interpretation = "ICR Probe is a positive closest-method baseline under both grouped OOF and strict transfer."
    elif pooled_gain > 0:
        interpretation = "ICR Probe is positive under grouped OOF but does not transfer directionally."
    else:
        interpretation = "The adapted ICR Probe does not improve over confidence on this protocol."
    config = rows[0].get("extractor_config") or {}
    report = {
        "schema_version": "icassp-icr-scifact-evaluation-v1",
        "rows": len(rows),
        "feature_layers": next(iter(layer_counts)),
        "upstream_commit": config.get("upstream_commit"),
        "all_document_grouped_oof": all_metrics,
        "dev_record_disjoint_transfer": strict_metrics,
        "interpretation": interpretation,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    args.output_md.write_text(render(report), encoding="utf-8")
    print(json.dumps({"pooled_gain": pooled_gain, "strict_gain": strict_gain, "interpretation": interpretation}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

