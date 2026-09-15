#!/usr/bin/env python3
"""Evaluate ECLT and optional ICR Probe on frozen VitaminC evidence pairs."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.metrics import average_precision_score, balanced_accuracy_score, roc_auc_score


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import eclt_features as eclt  # noqa: E402
import icr_probe as icr  # noqa: E402


SEED = 20260810
ECLT_GROUPS = {
    "confidence": "confidence",
    "final_relation": "last_layer",
    "relation_trajectory": "relation_trajectory",
    "full_trajectory": "trajectory",
}
PURE_RELATION_GROUPS = {
    "pure_final_relation": True,
    "pure_relation_trajectory": False,
}


def label(row: dict[str, Any]) -> int:
    return int((row.get("labels") or {}).get("supported", 0))


def pair_id(row: dict[str, Any]) -> str:
    return str((row.get("audit_metadata") or {}).get("pair_id") or row.get("record_id"))


def align_rows(
    eclt_path: Path, icr_path: Path | None
) -> tuple[list[dict[str, Any]], list[dict[str, Any]] | None]:
    e_rows = [row for row in eclt.iter_jsonl(eclt_path) if row.get("status") == "ok"]
    e_ids = [str(row.get("claim_id")) for row in e_rows]
    if len(e_ids) != len(set(e_ids)):
        raise ValueError("duplicate ECLT claim IDs")
    if icr_path is None:
        return e_rows, None
    i_by_id = {
        str(row.get("claim_id")): row
        for row in icr.iter_jsonl(icr_path)
        if row.get("status") == "ok"
    }
    if len(i_by_id) != len(e_rows):
        raise ValueError(f"unaligned row counts: ECLT={len(e_rows)} ICR={len(i_by_id)}")
    i_rows = []
    for row in e_rows:
        claim_id = str(row.get("claim_id"))
        other = i_by_id.get(claim_id)
        if other is None:
            raise ValueError(f"ICR missing claim {claim_id}")
        for key in ("record_id", "partition", "candidate_value", "labels", "audit_metadata"):
            if row.get(key) != other.get(key):
                raise ValueError(f"alignment mismatch claim={claim_id} key={key}")
        i_rows.append(other)
    return e_rows, i_rows


def metric(y: np.ndarray, probability: np.ndarray) -> dict[str, Any]:
    return {
        "n": int(len(y)),
        "positives": int(y.sum()),
        "roc_auc": float(roc_auc_score(y, probability)),
        "average_precision": float(average_precision_score(y, probability)),
        "balanced_accuracy": float(balanced_accuracy_score(y, probability >= 0.5)),
    }


def fit_eclt(
    train: list[dict[str, Any]], test: list[dict[str, Any]], feature_group: str
) -> np.ndarray:
    y = np.asarray([label(row) for row in train])
    model = eclt.make_model()
    model.fit([eclt.features(row, feature_group) for row in train], y)
    return model.predict_proba([eclt.features(row, feature_group) for row in test])[:, 1]


def pure_relation_features(row: dict[str, Any], last_only: bool) -> dict[str, float]:
    layers = sorted(
        ((row.get("features") or {}).get("layer_trajectory") or []),
        key=lambda item: item.get("layer", -1),
    )
    selected = layers[-1:] if last_only else layers
    values: dict[str, float] = {}
    for layer in selected:
        layer_id = int(layer.get("layer", -1))
        for key in eclt.RELATION_KEYS:
            eclt.put(values, f"L{layer_id}.{key}", layer.get(key))
    if not last_only:
        summary = (row.get("features") or {}).get("trajectory_summary") or {}
        for key in eclt.RELATION_KEYS:
            for statistic, value in (summary.get(key) or {}).items():
                eclt.put(values, f"trajectory.{key}.{statistic}", value)
    return values


def fit_pure_relation(
    train: list[dict[str, Any]], test: list[dict[str, Any]], last_only: bool
) -> np.ndarray:
    y = np.asarray([label(row) for row in train])
    model = eclt.make_model()
    model.fit([pure_relation_features(row, last_only) for row in train], y)
    return model.predict_proba(
        [pure_relation_features(row, last_only) for row in test]
    )[:, 1]


def pure_relation_oof(
    rows: list[dict[str, Any]], y: np.ndarray, groups: np.ndarray, last_only: bool
) -> np.ndarray:
    from sklearn.model_selection import StratifiedGroupKFold

    splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED)
    probability = np.full(len(rows), np.nan)
    values = [pure_relation_features(row, last_only) for row in rows]
    dummy = np.zeros((len(rows), 1))
    for train, test in splitter.split(dummy, y, groups):
        model = eclt.make_model()
        model.fit([values[index] for index in train], y[train])
        probability[test] = model.predict_proba(
            [values[index] for index in test]
        )[:, 1]
    if not np.isfinite(probability).all():
        raise AssertionError("incomplete pure-relation OOF probabilities")
    return probability


def transfer_predictions(
    e_train: list[dict[str, Any]],
    e_test: list[dict[str, Any]],
    i_train: list[dict[str, Any]] | None,
    i_test: list[dict[str, Any]] | None,
) -> dict[str, np.ndarray]:
    values = {
        name: fit_eclt(e_train, e_test, group) for name, group in ECLT_GROUPS.items()
    }
    values.update(
        {
            name: fit_pure_relation(e_train, e_test, last_only)
            for name, last_only in PURE_RELATION_GROUPS.items()
        }
    )
    if i_train is not None and i_test is not None:
        train_y = np.asarray([label(row) for row in i_train])
        values["icr_linear"] = icr.transfer(
            icr.matrix(i_train, "icr"), train_y, icr.matrix(i_test, "icr"), "linear"
        )
        values["icr_probe"] = icr.transfer(
            icr.matrix(i_train, "icr"), train_y, icr.matrix(i_test, "icr"), "mlp"
        )
    return values


def oof_predictions(
    e_rows: list[dict[str, Any]], i_rows: list[dict[str, Any]] | None
) -> dict[str, np.ndarray]:
    y = np.asarray([label(row) for row in e_rows])
    groups = np.asarray([str(row.get("record_id")) for row in e_rows])
    values = {
        name: eclt.grouped_oof(e_rows, y, groups, group)
        for name, group in ECLT_GROUPS.items()
    }
    values.update(
        {
            name: pure_relation_oof(e_rows, y, groups, last_only)
            for name, last_only in PURE_RELATION_GROUPS.items()
        }
    )
    if i_rows is not None:
        values["icr_linear"] = icr.grouped_oof(
            icr.matrix(i_rows, "icr"), y, groups, "linear"
        )
        values["icr_probe"] = icr.grouped_oof(
            icr.matrix(i_rows, "icr"), y, groups, "mlp"
        )
    return values


def paired_direction(
    rows: list[dict[str, Any]], probability: np.ndarray, bootstrap: int
) -> dict[str, Any]:
    groups: dict[str, list[int]] = defaultdict(list)
    for index, row in enumerate(rows):
        groups[pair_id(row)].append(index)
    outcomes: list[float] = []
    by_negative: dict[str, list[float]] = defaultdict(list)
    for indices in groups.values():
        if len(indices) != 2:
            raise ValueError(f"VitaminC pair does not contain two rows: {indices}")
        positive = [idx for idx in indices if label(rows[idx]) == 1]
        negative = [idx for idx in indices if label(rows[idx]) == 0]
        if len(positive) != 1 or len(negative) != 1:
            raise ValueError(f"invalid pair labels: {indices}")
        delta = float(probability[positive[0]] - probability[negative[0]])
        outcome = 1.0 if delta > 0 else 0.5 if delta == 0 else 0.0
        outcomes.append(outcome)
        negative_label = str((rows[negative[0]].get("audit_metadata") or {}).get("label"))
        by_negative[negative_label].append(outcome)
    values = np.asarray(outcomes, dtype=np.float64)
    rng = np.random.default_rng(SEED)
    sampled = np.empty(bootstrap, dtype=np.float64)
    for iteration in range(bootstrap):
        sampled[iteration] = rng.choice(values, size=len(values), replace=True).mean()
    return {
        "pairs": int(len(values)),
        "accuracy": float(values.mean()),
        "ci95_low": float(np.quantile(sampled, 0.025)),
        "ci95_high": float(np.quantile(sampled, 0.975)),
        "by_negative_label": {
            key: {"pairs": len(group), "accuracy": float(np.mean(group))}
            for key, group in sorted(by_negative.items())
        },
    }


def bootstrap_deltas(
    y: np.ndarray,
    values: dict[str, np.ndarray],
    groups: np.ndarray,
    bootstrap: int,
) -> dict[str, Any]:
    result = {
        "relation_minus_confidence": eclt.grouped_bootstrap_delta(
            y, values["confidence"], values["relation_trajectory"], groups, bootstrap
        ),
        "relation_minus_final": eclt.grouped_bootstrap_delta(
            y, values["final_relation"], values["relation_trajectory"], groups, bootstrap
        ),
        "full_minus_confidence": eclt.grouped_bootstrap_delta(
            y, values["confidence"], values["full_trajectory"], groups, bootstrap
        ),
        "pure_relation_minus_pure_final": eclt.grouped_bootstrap_delta(
            y,
            values["pure_final_relation"],
            values["pure_relation_trajectory"],
            groups,
            bootstrap,
        ),
    }
    if "icr_linear" in values:
        result["full_minus_icr_linear"] = eclt.grouped_bootstrap_delta(
            y, values["icr_linear"], values["full_trajectory"], groups, bootstrap
        )
        result["full_minus_icr_probe"] = eclt.grouped_bootstrap_delta(
            y, values["icr_probe"], values["full_trajectory"], groups, bootstrap
        )
        result["pure_relation_minus_icr_linear"] = eclt.grouped_bootstrap_delta(
            y,
            values["icr_linear"],
            values["pure_relation_trajectory"],
            groups,
            bootstrap,
        )
    return result


def evaluate_block(
    rows: list[dict[str, Any]], values: dict[str, np.ndarray], bootstrap: int
) -> dict[str, Any]:
    y = np.asarray([label(row) for row in rows])
    groups = np.asarray([str(row.get("record_id")) for row in rows])
    return {
        "metrics": {name: metric(y, probability) for name, probability in values.items()},
        "paired_direction": {
            name: paired_direction(rows, probability, bootstrap)
            for name, probability in values.items()
        },
        "auc_deltas": bootstrap_deltas(y, values, groups, bootstrap),
    }


def render(report: dict[str, Any]) -> str:
    lines = [
        "# ECLT VitaminC Contrastive Transfer",
        "",
        f"- rows/pairs: {report['rows']}/{report['pairs']}",
        f"- official train/test rows: {report['partition_counts']['train']}/"
        f"{report['partition_counts']['test']}",
        "- construction: real revisions, one case per pair, identical claim with supporting and contrast evidence",
        "- official development split: unused",
        "",
    ]
    for title, key in (
        ("Official train-to-test transfer", "official_test_transfer"),
        ("All rows, case-grouped OOF", "all_case_grouped_oof"),
    ):
        block = report[key]
        lines.extend(
            [
                f"## {title}",
                "",
                "| Method | ROC AUC | AP | Balanced accuracy | Paired direction | 95% CI |",
                "|---|---:|---:|---:|---:|---:|",
            ]
        )
        for name, values in block["metrics"].items():
            pair = block["paired_direction"][name]
            lines.append(
                f"| {name} | {values['roc_auc']:.4f} | {values['average_precision']:.4f} | "
                f"{values['balanced_accuracy']:.4f} | {pair['accuracy']:.4f} | "
                f"[{pair['ci95_low']:.4f}, {pair['ci95_high']:.4f}] |"
            )
        lines.extend(["", f"Grouped-bootstrap AUC deltas: `{block['auc_deltas']}`", ""])
    lines.extend(["## Decision rule", "", f"- {report['verdict']}"])
    lines.extend(f"- {item}" for item in report["interpretation"])
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--eclt-features", type=Path, required=True)
    parser.add_argument("--icr-features", type=Path)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-md", type=Path, required=True)
    parser.add_argument("--expected-rows", type=int, default=4000)
    parser.add_argument("--bootstrap", type=int, default=2000)
    args = parser.parse_args()

    e_rows, i_rows = align_rows(args.eclt_features, args.icr_features)
    if len(e_rows) != args.expected_rows:
        raise ValueError(f"expected {args.expected_rows} valid ECLT rows, got {len(e_rows)}")
    train_idx = [idx for idx, row in enumerate(e_rows) if row.get("partition") == "train"]
    test_idx = [idx for idx, row in enumerate(e_rows) if row.get("partition") == "test"]
    if not train_idx or not test_idx:
        raise ValueError("VitaminC train/test partitions missing")
    e_train = [e_rows[idx] for idx in train_idx]
    e_test = [e_rows[idx] for idx in test_idx]
    i_train = [i_rows[idx] for idx in train_idx] if i_rows is not None else None
    i_test = [i_rows[idx] for idx in test_idx] if i_rows is not None else None
    train_cases = {str(row.get("record_id")) for row in e_train}
    test_cases = {str(row.get("record_id")) for row in e_test}
    if train_cases & test_cases:
        raise ValueError("VitaminC train/test case overlap")

    transfer_values = transfer_predictions(e_train, e_test, i_train, i_test)
    oof_values = oof_predictions(e_rows, i_rows)
    transfer_block = evaluate_block(e_test, transfer_values, args.bootstrap)
    all_block = evaluate_block(e_rows, oof_values, args.bootstrap)

    full_delta = transfer_block["auc_deltas"]["full_minus_confidence"]
    relation_delta = transfer_block["auc_deltas"]["pure_relation_minus_pure_final"]
    if full_delta["ci95_low"] > 0 and relation_delta["ci95_low"] > 0:
        verdict = "PUBLIC_CONTRASTIVE_TRANSFER_POSITIVE"
    elif full_delta["mean"] > 0 and relation_delta["mean"] > 0:
        verdict = "PUBLIC_CONTRASTIVE_TRANSFER_DIRECTIONAL"
    else:
        verdict = "PUBLIC_CONTRASTIVE_TRANSFER_NULL_OR_NEGATIVE"

    kernel_audit = None
    if i_rows is not None:
        e_config = e_rows[0].get("extractor_config") or {}
        i_config = i_rows[0].get("extractor_config") or {}
        kernel_audit = {
            "eclt_attention_implementation": e_config.get("attention_implementation"),
            "icr_attention_implementation": i_config.get("attention_implementation"),
            "equal_prompt_token_rows": sum(
                left["features"]["prompt_tokens"] == right["features"]["prompt_tokens"]
                for left, right in zip(e_rows, i_rows)
            ),
            "equal_target_token_rows": sum(
                left["features"]["target_tokens"] == right["features"]["target_tokens"]
                for left, right in zip(e_rows, i_rows)
            ),
        }
    report = {
        "schema_version": "icassp-eclt-vitaminc-evaluation-v1",
        "rows": len(e_rows),
        "pairs": len({pair_id(row) for row in e_rows}),
        "partition_counts": Counter(str(row.get("partition")) for row in e_rows),
        "negative_label_counts": Counter(
            str((row.get("audit_metadata") or {}).get("label"))
            for row in e_rows
            if label(row) == 0
        ),
        "train_test_case_overlap": 0,
        "official_test_transfer": transfer_block,
        "all_case_grouped_oof": all_block,
        "kernel_audit": kernel_audit,
        "verdict": verdict,
        "interpretation": [
            "The official test set is the primary result; pooled OOF is secondary.",
            "Paired direction asks whether the same claim scores higher with its supporting evidence than with its contrast evidence.",
            "A positive relation-over-final result rejects the explanation that a final-layer claim embedding alone is sufficient.",
        ],
    }
    # Convert Counters for stable JSON serialization.
    report["partition_counts"] = dict(report["partition_counts"])
    report["negative_label_counts"] = dict(report["negative_label_counts"])
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    args.output_md.write_text(render(report), encoding="utf-8")
    print(json.dumps({"verdict": verdict, "official_test": transfer_block["metrics"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
