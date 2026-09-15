#!/usr/bin/env python3
"""Evaluate whether ECLT benefits from layer depth and ordered trajectories."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Any, Callable

import numpy as np
from sklearn.model_selection import StratifiedGroupKFold

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import eclt_features as base  # noqa: E402


FeatureFn = Callable[[dict[str, Any]], dict[str, float]]
RELATION_KEYS = base.RELATION_KEYS


def finite(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def selected_indices(count: int, requested: int) -> list[int]:
    if requested <= 1:
        return [count - 1]
    return sorted({int(round(value)) for value in np.linspace(0, count - 1, requested)})


def ordered_layers(row: dict[str, Any], requested: int, shuffled: bool = False) -> list[dict[str, Any]]:
    layers = sorted(
        (row.get("features") or {}).get("layer_trajectory") or [],
        key=lambda item: int(item.get("layer", -1)),
    )
    if not layers:
        return []
    selected = [layers[index] for index in selected_indices(len(layers), requested)]
    if shuffled:
        digest = hashlib.sha256(str(row.get("claim_id")).encode("utf-8")).digest()
        rng = np.random.default_rng(int.from_bytes(digest[:8], "big"))
        order = rng.permutation(len(selected))
        selected = [selected[int(index)] for index in order]
    return selected


def add_summary(out: dict[str, float], layers: list[dict[str, Any]], *, ordered: bool) -> None:
    for key in RELATION_KEYS:
        values = [finite(layer.get(key)) for layer in layers]
        values = [value for value in values if value is not None]
        if not values:
            continue
        array = np.asarray(values, dtype=float)
        out[f"summary.{key}.mean"] = float(array.mean())
        out[f"summary.{key}.std"] = float(array.std())
        out[f"summary.{key}.range"] = float(array.max() - array.min())
        if ordered:
            out[f"summary.{key}.first"] = float(array[0])
            out[f"summary.{key}.last"] = float(array[-1])
            if len(array) >= 2:
                x = np.linspace(0.0, 1.0, len(array))
                out[f"summary.{key}.slope"] = float(np.polyfit(x, array, 1)[0])
                out[f"summary.{key}.auc"] = float(np.trapz(array, x))


def relation_features(
    row: dict[str, Any], requested: int, *, shuffled: bool = False, orderless: bool = False
) -> dict[str, float]:
    layers = ordered_layers(row, requested, shuffled=shuffled)
    out: dict[str, float] = {}
    if not orderless:
        for position, layer in enumerate(layers):
            for key in RELATION_KEYS:
                value = finite(layer.get(key))
                if value is not None:
                    out[f"P{position}.{key}"] = value
    add_summary(out, layers, ordered=not orderless)
    return out


def confidence_features(row: dict[str, Any]) -> dict[str, float]:
    out: dict[str, float] = {}
    for key, value in ((row.get("features") or {}).get("confidence") or {}).items():
        number = finite(value)
        if number is not None:
            out[f"confidence.{key}"] = number
    return out


def full_features(row: dict[str, Any]) -> dict[str, float]:
    return base.features(row, "trajectory")


FEATURES: dict[str, FeatureFn] = {
    "confidence": confidence_features,
    "final_relation": lambda row: relation_features(row, 1),
    "ordered_3": lambda row: relation_features(row, 3),
    "ordered_5": lambda row: relation_features(row, 5),
    "ordered_9": lambda row: relation_features(row, 9),
    "orderless_9": lambda row: relation_features(row, 9, orderless=True),
    "shuffled_9": lambda row: relation_features(row, 9, shuffled=True),
    "full_eclt": full_features,
}


def grouped_oof(
    rows: list[dict[str, Any]], labels: np.ndarray, groups: np.ndarray, feature_fn: FeatureFn
) -> np.ndarray:
    splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=20260807)
    probabilities = np.full(len(rows), np.nan)
    values = [feature_fn(row) for row in rows]
    dummy = np.zeros((len(rows), 1))
    for train, test in splitter.split(dummy, labels, groups):
        model = base.make_model()
        model.fit([values[index] for index in train], labels[train])
        probabilities[test] = model.predict_proba([values[index] for index in test])[:, 1]
    if not np.all(np.isfinite(probabilities)):
        raise AssertionError("incomplete grouped OOF predictions")
    return probabilities


def transfer(
    train: list[dict[str, Any]], test: list[dict[str, Any]], train_y: np.ndarray, feature_fn: FeatureFn
) -> np.ndarray:
    model = base.make_model()
    model.fit([feature_fn(row) for row in train], train_y)
    return model.predict_proba([feature_fn(row) for row in test])[:, 1]


def evaluate_model(path: Path, bootstrap: int) -> dict[str, Any]:
    rows = [row for row in base.iter_jsonl(path) if row.get("status") == "ok"]
    if len(rows) != 773 or len({str(row.get("claim_id")) for row in rows}) != 773:
        raise ValueError(f"expected 773 unique rows: {path}")
    labels = np.asarray([int((row.get("labels") or {}).get("supported", 0)) for row in rows])
    groups = np.asarray([str(row.get("record_id")) for row in rows])
    predictions = {name: grouped_oof(rows, labels, groups, fn) for name, fn in FEATURES.items()}

    train = [row for row in rows if row.get("partition") == "train"]
    train_records = {str(row.get("record_id")) for row in train}
    test = [
        row
        for row in rows
        if row.get("partition") == "dev" and str(row.get("record_id")) not in train_records
    ]
    train_y = np.asarray([int((row.get("labels") or {}).get("supported", 0)) for row in train])
    test_y = np.asarray([int((row.get("labels") or {}).get("supported", 0)) for row in test])
    test_groups = np.asarray([str(row.get("record_id")) for row in test])
    transfer_predictions = {
        name: transfer(train, test, train_y, fn) for name, fn in FEATURES.items()
    }

    def deltas(y: np.ndarray, values: dict[str, np.ndarray], group_values: np.ndarray) -> dict[str, Any]:
        return {
            "ordered_9_minus_final": base.grouped_bootstrap_delta(
                y, values["final_relation"], values["ordered_9"], group_values, bootstrap
            ),
            "ordered_9_minus_orderless": base.grouped_bootstrap_delta(
                y, values["orderless_9"], values["ordered_9"], group_values, bootstrap
            ),
            "ordered_9_minus_shuffled": base.grouped_bootstrap_delta(
                y, values["shuffled_9"], values["ordered_9"], group_values, bootstrap
            ),
            "ordered_9_minus_ordered_3": base.grouped_bootstrap_delta(
                y, values["ordered_3"], values["ordered_9"], group_values, bootstrap
            ),
        }

    return {
        "source": str(path),
        "attention_implementation": (rows[0].get("extractor_config") or {}).get(
            "attention_implementation", "sdpa"
        ),
        "grouped_oof": {
            "metrics": {name: base.score(labels, value) for name, value in predictions.items()},
            "deltas": deltas(labels, predictions, groups),
        },
        "record_disjoint": {
            "rows": len(test),
            "metrics": {
                name: base.score(test_y, value) for name, value in transfer_predictions.items()
            },
            "deltas": deltas(test_y, transfer_predictions, test_groups),
        },
    }


def render(report: dict[str, Any]) -> str:
    lines = [
        "# ECLT Layer-Depth and Order Ablation",
        "",
        "All feature sets use the same claims, labels, record groups, classifier, and folds.",
        "`shuffled_9` preserves feature count and per-row layer values but randomizes their depth positions.",
        "",
    ]
    for model, result in report["models"].items():
        lines.extend([f"## {model}", "", f"- attention: `{result['attention_implementation']}`", ""])
        for title, key in (("Document-grouped OOF", "grouped_oof"), ("Record-disjoint transfer", "record_disjoint")):
            block = result[key]
            lines.extend(
                [
                    f"### {title}",
                    "",
                    "| Feature set | ROC AUC | AP | Balanced accuracy |",
                    "|---|---:|---:|---:|",
                ]
            )
            for name in FEATURES:
                metric = block["metrics"][name]
                lines.append(
                    f"| {name} | {metric['roc_auc']:.4f} | {metric['average_precision']:.4f} | "
                    f"{metric['balanced_accuracy']:.4f} |"
                )
            lines.extend(["", f"Grouped-bootstrap deltas: `{block['deltas']}`", ""])
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", action="append", required=True, help="MODEL=JSONL")
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-md", type=Path, required=True)
    parser.add_argument("--bootstrap", type=int, default=2000)
    args = parser.parse_args()
    sources: dict[str, Path] = {}
    for item in args.features:
        if "=" not in item:
            raise ValueError(f"invalid --features value: {item}")
        model, path = item.split("=", 1)
        sources[model] = Path(path)
    report = {
        "schema_version": "icassp-eclt-depth-ablation-v1",
        "models": {name: evaluate_model(path, args.bootstrap) for name, path in sources.items()},
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    args.output_md.write_text(render(report), encoding="utf-8")
    print(json.dumps({name: result["grouped_oof"]["deltas"] for name, result in report["models"].items()}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
