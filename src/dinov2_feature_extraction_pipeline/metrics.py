"""Classification metrics over a labelled-image dataset, the majority baseline and a k-NN baseline.

Predictions are compared with gold labels: **accuracy**, **macro-F1** (the unweighted mean of per-class F1, so
a rare class counts as much as a common one), per-class precision / recall / F1 / support and the confusion
matrix. The **majority baseline** predicts the most frequent training label for every image; the **k-NN
baseline** labels each test image by a cosine vote of its `k` nearest training images in the frozen feature
space — what the checkpoint's representation gives with no training at all.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np

METRIC_DEFINITIONS = {
    "accuracy": "fraction of images whose predicted label equals the gold label",
    "macro_f1": "unweighted mean over classes of the per-class F1 (precision-recall harmonic mean)",
    "per_class": "precision, recall, F1 and support for every class of the gold label set",
    "confusion": "rows are gold classes, columns predicted classes, in the order of `classes`",
    "log_loss": "mean negative log-probability the head assigns to the gold label (the selection signal)",
}


def classification_metrics(
    y_true: Sequence[str], y_pred: Sequence[str], classes: Sequence[str]
) -> dict[str, Any]:
    """Accuracy, macro-F1, per-class scores and the confusion matrix over a fixed class order."""
    if not y_true:
        raise ValueError("no images to score")
    if len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred must align")
    order = list(classes)
    index = {c: i for i, c in enumerate(order)}
    unknown = [y for y in list(y_true) + list(y_pred) if y not in index]
    if unknown:
        raise ValueError(f"label {unknown[0]!r} is not in classes")
    confusion = np.zeros((len(order), len(order)), dtype=np.int64)
    for t, p in zip(y_true, y_pred, strict=True):
        confusion[index[t], index[p]] += 1
    per_class = {}
    f1s = []
    for i, name in enumerate(order):
        tp = int(confusion[i, i])
        fp = int(confusion[:, i].sum() - tp)
        fn = int(confusion[i, :].sum() - tp)
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        per_class[name] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": int(confusion[i, :].sum()),
        }
        f1s.append(f1)
    return {
        "n": len(y_true),
        "accuracy": float(np.trace(confusion) / len(y_true)),
        "macro_f1": float(sum(f1s) / len(f1s)),
        "per_class": per_class,
        "confusion": confusion.tolist(),
        "classes": order,
        "definitions": dict(METRIC_DEFINITIONS),
    }


def majority_baseline(
    train_labels: Sequence[str], test_labels: Sequence[str], classes: Sequence[str]
) -> dict[str, Any]:
    """Predict the most frequent training label (ties by class order) for every test image."""
    if not train_labels:
        raise ValueError("no training labels")
    counts = {c: 0 for c in classes}
    for label in train_labels:
        counts[label] = counts.get(label, 0) + 1
    majority = max(classes, key=lambda c: (counts.get(c, 0), -list(classes).index(c)))
    result = classification_metrics(test_labels, [majority] * len(test_labels), classes)
    result["baseline"] = f"majority training label ({majority!r}) predicted for every image"
    return result


def knn_predict(
    train_features: np.ndarray, train_labels: Sequence[str], test_features: np.ndarray, *, k: int = 5
) -> list[str]:
    """Cosine k-NN vote (unit-normalised rows assumed; normalised here anyway); ties by summed similarity."""
    if k < 1:
        raise ValueError("k must be positive")
    x = np.asarray(train_features, dtype=np.float32)
    q = np.asarray(test_features, dtype=np.float32)
    if x.ndim != 2 or q.ndim != 2 or x.shape[1] != q.shape[1]:
        raise ValueError("features must be 2-D with the same width")
    if len(train_labels) != x.shape[0]:
        raise ValueError("train_labels must align with train_features")
    x = x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-12)
    q = q / np.maximum(np.linalg.norm(q, axis=1, keepdims=True), 1e-12)
    sims = q @ x.T
    k = min(k, x.shape[0])
    out = []
    labels = list(train_labels)
    for row in sims:
        top = np.argsort(-row, kind="stable")[:k]
        votes: dict[str, float] = {}
        for j in top:
            votes[labels[j]] = votes.get(labels[j], 0.0) + float(row[j])
        out.append(max(votes, key=lambda c: (votes[c], -labels.index(c))))
    return out


def knn_baseline(
    train_features: np.ndarray,
    train_labels: Sequence[str],
    test_features: np.ndarray,
    test_labels: Sequence[str],
    classes: Sequence[str],
    *,
    k: int = 5,
) -> dict[str, Any]:
    """k-NN over frozen features — what the representation gives without any training."""
    result = classification_metrics(
        test_labels, knn_predict(train_features, train_labels, test_features, k=k), classes
    )
    result["baseline"] = f"cosine {k}-NN vote over the frozen features of the training images"
    result["k"] = k
    return result
