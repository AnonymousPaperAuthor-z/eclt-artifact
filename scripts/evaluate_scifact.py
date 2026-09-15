#!/usr/bin/env python3
"""Evaluate ECLT on SciFact with document-grouped OOF and official dev transfer."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

from eclt_features import (
    FEATURE_GROUPS,
    features,
    grouped_bootstrap_delta,
    grouped_oof,
    iter_jsonl,
    make_model,
    score,
)


def labels(rows: list[dict[str, Any]]) -> np.ndarray:
    return np.asarray([int((row.get("labels") or {}).get("supported", 0)) for row in rows])


def transfer_predictions(
    train: list[dict[str, Any]],
    test: list[dict[str, Any]],
) -> tuple[dict[str, dict[str, Any]], dict[str, np.ndarray]]:
    train_y = labels(train)
    test_y = labels(test)
    metrics = {}
    predictions = {}
    for group in FEATURE_GROUPS:
        model = make_model()
        model.fit([features(row, group) for row in train], train_y)
        probability = model.predict_proba([features(row, group) for row in test])[:, 1]
        metrics[group] = score(test_y, probability)
        predictions[group] = probability
    return metrics, predictions


def bootstrap_deltas(
    y: np.ndarray,
    predictions: dict[str, np.ndarray],
    groups: np.ndarray,
    iterations: int,
) -> dict[str, Any]:
    return {
        "relation_minus_confidence": grouped_bootstrap_delta(
            y, predictions["confidence"], predictions["relation_trajectory"], groups, iterations
        ),
        "relation_minus_last_layer": grouped_bootstrap_delta(
            y, predictions["last_layer"], predictions["relation_trajectory"], groups, iterations
        ),
        "trajectory_minus_confidence": grouped_bootstrap_delta(
            y, predictions["confidence"], predictions["trajectory"], groups, iterations
        ),
    }


def render(report: dict[str, Any]) -> str:
    lines = [
        "# ECLT SciFact Public-Transfer Evaluation",
        "",
        f"- rows: {report['rows']}",
        f"- train/dev/disjoint-dev: {report['partition_counts']}",
        f"- train/dev document overlap: {report['train_dev_record_overlap']}",
        "- model input: candidate claim plus complete title and abstract",
        "- excluded from model input: gold label and rationale sentence indices",
        "",
        "## All-label document-grouped OOF",
        "",
        "| Feature family | ROC AUC | AP | Balanced accuracy |",
        "|---|---:|---:|---:|",
    ]
    for name in FEATURE_GROUPS:
        metric = report["all_document_grouped_oof"][name]
        lines.append(
            f"| {name} | {metric['roc_auc']:.4f} | {metric['average_precision']:.4f} | "
            f"{metric['balanced_accuracy']:.4f} |"
        )
    lines.extend(
        [
            "",
            f"Bootstrap deltas: `{report['all_document_grouped_oof']['bootstrap_deltas']}`",
            "",
        "## Document-grouped train OOF",
        "",
        "| Feature family | ROC AUC | AP | Balanced accuracy |",
        "|---|---:|---:|---:|",
        ]
    )
    for name in FEATURE_GROUPS:
        metric = report["train_document_grouped_oof"][name]
        lines.append(
            f"| {name} | {metric['roc_auc']:.4f} | {metric['average_precision']:.4f} | "
            f"{metric['balanced_accuracy']:.4f} |"
        )
    lines.extend(
        [
            "",
            f"Bootstrap deltas: `{report['train_document_grouped_oof']['bootstrap_deltas']}`",
            "",
            "## Official dev transfer",
            "",
            "The headline uses only dev documents absent from training.",
            "",
            "| Feature family | ROC AUC | AP | Balanced accuracy |",
            "|---|---:|---:|---:|",
        ]
    )
    for name in FEATURE_GROUPS:
        metric = report["dev_record_disjoint_transfer"][name]
        lines.append(
            f"| {name} | {metric['roc_auc']:.4f} | {metric['average_precision']:.4f} | "
            f"{metric['balanced_accuracy']:.4f} |"
        )
    lines.extend(
        [
            "",
            f"Bootstrap deltas: `{report['dev_record_disjoint_transfer']['bootstrap_deltas']}`",
            "",
            "## Decision rule",
            "",
        ]
    )
    lines.extend(f"- {item}" for item in report["interpretation"])
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-md", type=Path, required=True)
    parser.add_argument("--bootstrap", type=int, default=2000)
    args = parser.parse_args()

    rows = [row for row in iter_jsonl(args.features) if row.get("status") == "ok"]
    if not rows:
        raise ValueError("no valid SciFact feature rows")
    invalid_prompt = [
        row.get("claim_id")
        for row in rows
        if ((row.get("extractor_config") or {}).get("prompt_mode") != "claim_verification")
    ]
    if invalid_prompt:
        raise ValueError(f"non-SciFact prompt mode: {invalid_prompt[:5]}")

    train = [row for row in rows if row.get("partition") == "train"]
    dev = [row for row in rows if row.get("partition") == "dev"]
    train_records = {str(row.get("record_id")) for row in train}
    dev_disjoint = [row for row in dev if str(row.get("record_id")) not in train_records]
    if len(train) < 100 or len(dev_disjoint) < 50:
        raise ValueError(f"insufficient rows train={len(train)} disjoint_dev={len(dev_disjoint)}")

    train_y = labels(train)
    train_groups = np.asarray([str(row.get("record_id")) for row in train])
    train_predictions = {
        name: grouped_oof(train, train_y, train_groups, name) for name in FEATURE_GROUPS
    }
    train_metrics = {name: score(train_y, values) for name, values in train_predictions.items()}
    train_metrics["bootstrap_deltas"] = bootstrap_deltas(
        train_y, train_predictions, train_groups, args.bootstrap
    )

    all_y = labels(rows)
    all_groups = np.asarray([str(row.get("record_id")) for row in rows])
    all_predictions = {
        name: grouped_oof(rows, all_y, all_groups, name) for name in FEATURE_GROUPS
    }
    all_metrics = {name: score(all_y, values) for name, values in all_predictions.items()}
    all_metrics["bootstrap_deltas"] = bootstrap_deltas(
        all_y, all_predictions, all_groups, args.bootstrap
    )

    dev_metrics, dev_predictions = transfer_predictions(train, dev_disjoint)
    dev_y = labels(dev_disjoint)
    dev_groups = np.asarray([str(row.get("record_id")) for row in dev_disjoint])
    dev_metrics["bootstrap_deltas"] = bootstrap_deltas(
        dev_y, dev_predictions, dev_groups, args.bootstrap
    )

    all_dev_metrics, _ = transfer_predictions(train, dev)
    relation_gain = (
        dev_metrics["relation_trajectory"]["roc_auc"] - dev_metrics["confidence"]["roc_auc"]
    )
    relation_last_gain = (
        dev_metrics["relation_trajectory"]["roc_auc"] - dev_metrics["last_layer"]["roc_auc"]
    )
    relation_ci = dev_metrics["bootstrap_deltas"]["relation_minus_confidence"]
    last_ci = dev_metrics["bootstrap_deltas"]["relation_minus_last_layer"]
    if relation_ci["ci95_low"] > 0 and last_ci["ci95_low"] > 0:
        verdict = "PUBLIC_TRANSFER_POSITIVE"
        wording = "Layer relations transfer beyond confidence and the last layer on record-disjoint SciFact."
    elif relation_gain > 0 and relation_last_gain > 0:
        verdict = "PUBLIC_TRANSFER_DIRECTIONAL"
        wording = "Point estimates favor layer relations, but uncertainty does not support a strong transfer claim."
    else:
        verdict = "PUBLIC_TRANSFER_NULL_OR_NEGATIVE"
        wording = "The public task does not support a general cross-domain layer-trajectory claim."

    report = {
        "schema_version": "icassp-eclt-scifact-evaluation-v1",
        "rows": len(rows),
        "partition_counts": {
            "train": len(train),
            "dev": len(dev),
            "dev_record_disjoint": len(dev_disjoint),
        },
        "label_counts": dict(
            Counter(
                f"{row.get('partition')}::{int((row.get('labels') or {}).get('supported', 0))}"
                for row in rows
            )
        ),
        "record_counts": {
            "train": len(train_records),
            "dev": len({str(row.get('record_id')) for row in dev}),
            "dev_record_disjoint": len(set(dev_groups)),
        },
        "train_dev_record_overlap": len(
            train_records & {str(row.get("record_id")) for row in dev}
        ),
        "train_document_grouped_oof": train_metrics,
        "all_document_grouped_oof": all_metrics,
        "dev_record_disjoint_transfer": dev_metrics,
        "dev_all_transfer_diagnostic": all_dev_metrics,
        "verdict": verdict,
        "interpretation": [
            wording,
            f"Record-disjoint dev relation gain over confidence: {relation_gain:+.4f} ROC AUC.",
            f"Record-disjoint dev relation gain over last layer: {relation_last_gain:+.4f} ROC AUC.",
            "Only a positive record-disjoint result may support a cross-domain claim; otherwise SciFact remains a reported boundary condition.",
        ],
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    args.output_md.write_text(render(report), encoding="utf-8")
    print(
        json.dumps(
            {
                "verdict": verdict,
                "record_disjoint_dev": len(dev_disjoint),
                "relation_auc": dev_metrics["relation_trajectory"]["roc_auc"],
                "confidence_auc": dev_metrics["confidence"]["roc_auc"],
                "last_layer_auc": dev_metrics["last_layer"]["roc_auc"],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
