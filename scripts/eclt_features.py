"""Feature definitions and grouped evaluation for ECLT."""

from __future__ import annotations

import argparse

import hashlib

import json

import math

import re

from collections import Counter, defaultdict

from pathlib import Path

from typing import Any, Iterable

import numpy as np

from sklearn.feature_extraction import DictVectorizer

from sklearn.impute import SimpleImputer

from sklearn.linear_model import LogisticRegression

from sklearn.metrics import average_precision_score, balanced_accuracy_score, roc_auc_score

from sklearn.model_selection import StratifiedGroupKFold

from sklearn.pipeline import Pipeline

from sklearn.preprocessing import StandardScaler

RELATION_KEYS = (
    "predictor_evidence_cos",
    "candidate_evidence_cos",
    "predictor_context_cos",
    "candidate_context_cos",
    "predictor_evidence_l2",
    "candidate_evidence_l2",
)

SCALAR_KEYS = RELATION_KEYS + (
    "predictor_norm",
    "candidate_norm",
    "evidence_norm",
)

FEATURE_GROUPS = (
    "lexical",
    "confidence",
    "last_layer",
    "relation_trajectory",
    "trajectory",
)

def iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                row = json.loads(line)
                if isinstance(row, dict):
                    yield row

def finite(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(number):
        return None
    return max(-10_000.0, min(10_000.0, number))

def put(target: dict[str, float], key: str, value: Any) -> None:
    number = finite(value)
    if number is not None:
        target[key] = number

def features(row: dict[str, Any], group: str) -> dict[str, float]:
    source = row.get("features") or {}
    out = {
        "span_present": float(bool(source.get("evidence_span_present"))),
        "occurrence_count": float(source.get("evidence_occurrence_count") or 0),
        "prompt_tokens": float(source.get("prompt_tokens") or 0),
        "target_tokens": float(source.get("target_tokens") or 0),
        "context_truncated": float(bool(source.get("context_truncated"))),
    }
    if group == "lexical":
        return out
    for key, value in (source.get("confidence") or {}).items():
        put(out, f"confidence.{key}", value)
    if group == "confidence":
        return out
    layers = sorted(source.get("layer_trajectory") or [], key=lambda item: item.get("layer", -1))
    selected = layers[-1:] if group == "last_layer" else layers
    scalar_keys = RELATION_KEYS if group == "relation_trajectory" else SCALAR_KEYS
    for layer in selected:
        layer_id = int(layer.get("layer", -1))
        for key in scalar_keys:
            put(out, f"L{layer_id}.{key}", layer.get(key))
    if group == "last_layer":
        return out
    for key, values in (source.get("trajectory_summary") or {}).items():
        for statistic, value in (values or {}).items():
            put(out, f"trajectory.{key}.{statistic}", value)
    return out

def make_model() -> Pipeline:
    return Pipeline(
        [
            ("vectorizer", DictVectorizer(sparse=False)),
            ("imputer", SimpleImputer(strategy="constant", fill_value=0.0)),
            ("scaler", StandardScaler()),
            (
                "classifier",
                LogisticRegression(
                    C=0.1,
                    class_weight="balanced",
                    max_iter=3000,
                    solver="liblinear",
                    random_state=20260807,
                ),
            ),
        ]
    )

def score(y: np.ndarray, probability: np.ndarray) -> dict[str, Any]:
    if len(np.unique(y)) < 2:
        return {"n": int(len(y)), "positives": int(np.sum(y)), "roc_auc": None}
    return {
        "n": int(len(y)),
        "positives": int(np.sum(y)),
        "roc_auc": float(roc_auc_score(y, probability)),
        "average_precision": float(average_precision_score(y, probability)),
        "balanced_accuracy": float(balanced_accuracy_score(y, probability >= 0.5)),
    }

def grouped_oof(
    rows: list[dict[str, Any]],
    labels: np.ndarray,
    groups: np.ndarray,
    feature_group: str,
) -> np.ndarray:
    unique_groups = np.unique(groups)
    if len(unique_groups) < 5:
        raise ValueError(f"insufficient groups for {feature_group}: {len(unique_groups)}")
    splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=20260807)
    probability = np.full(len(rows), np.nan)
    dummy = np.zeros((len(rows), 1))
    values = [features(row, feature_group) for row in rows]
    for train, test in splitter.split(dummy, labels, groups):
        model = make_model()
        model.fit([values[idx] for idx in train], labels[train])
        probability[test] = model.predict_proba([values[idx] for idx in test])[:, 1]
    if not np.all(np.isfinite(probability)):
        raise AssertionError(f"incomplete OOF predictions for {feature_group}")
    return probability

def grouped_bootstrap_delta(
    labels: np.ndarray,
    baseline: np.ndarray,
    proposed: np.ndarray,
    groups: np.ndarray,
    iterations: int,
) -> dict[str, Any]:
    unique_groups = np.unique(groups)
    indices = {group: np.flatnonzero(groups == group) for group in unique_groups}
    rng = np.random.default_rng(20260807)
    deltas = []
    for _ in range(iterations):
        sampled = rng.choice(unique_groups, size=len(unique_groups), replace=True)
        selected = np.concatenate([indices[group] for group in sampled])
        if len(np.unique(labels[selected])) < 2:
            continue
        deltas.append(
            roc_auc_score(labels[selected], proposed[selected])
            - roc_auc_score(labels[selected], baseline[selected])
        )
    values = np.asarray(deltas)
    return {
        "iterations": int(len(values)),
        "mean": float(np.mean(values)),
        "ci95_low": float(np.quantile(values, 0.025)),
        "ci95_high": float(np.quantile(values, 0.975)),
    }
