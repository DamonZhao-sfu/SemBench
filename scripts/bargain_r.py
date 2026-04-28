#!/usr/bin/env python3
from __future__ import annotations

import argparse
from dataclasses import dataclass
from math import sqrt
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class BargainRConfig:
    recall_target: float
    delta: float
    learning_rate: float = 0.1
    epochs: int = 400
    l2: float = 1e-4
    seed: int = 7


@dataclass(frozen=True)
class ThresholdResult:
    threshold: float
    coverage: float
    recall_empirical: float
    recall_lower_bound: float


class LogisticRegressionProxy:
    def __init__(self, learning_rate: float, epochs: int, l2: float, seed: int) -> None:
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.l2 = l2
        self.seed = seed
        self.weights: np.ndarray | None = None
        self.bias: float = 0.0

    def fit(self, x: np.ndarray, y: np.ndarray) -> None:
        rng = np.random.default_rng(self.seed)
        n_samples, n_features = x.shape
        w = rng.normal(0.0, 0.01, size=n_features)
        b = 0.0
        for _ in range(self.epochs):
            logits = x @ w + b
            probs = 1.0 / (1.0 + np.exp(-np.clip(logits, -40.0, 40.0)))
            error = probs - y
            grad_w = (x.T @ error) / n_samples + self.l2 * w
            grad_b = float(np.mean(error))
            w -= self.learning_rate * grad_w
            b -= self.learning_rate * grad_b
        self.weights = w
        self.bias = b

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        if self.weights is None:
            raise RuntimeError("Proxy is not fitted.")
        logits = x @ self.weights + self.bias
        return 1.0 / (1.0 + np.exp(-np.clip(logits, -40.0, 40.0)))

    def predict(self, x: np.ndarray) -> np.ndarray:
        probs = self.predict_proba(x)
        return (probs >= 0.5).astype(int)


class CorrectSmoothOracle:
    def __init__(self, smoothing_alpha: float = 0.5) -> None:
        self.smoothing_alpha = smoothing_alpha

    def predict(self, raw_score: np.ndarray) -> np.ndarray:
        smoothed = self.score(raw_score)
        return (smoothed >= 0.5).astype(int)

    def score(self, raw_score: np.ndarray) -> np.ndarray:
        return self.smoothing_alpha * raw_score + (1.0 - self.smoothing_alpha) * 0.5



def lr_confidence(probs: np.ndarray) -> np.ndarray:
    return np.maximum(probs, 1.0 - probs)


def wilson_lower_bound(successes: int, total: int, alpha: float) -> float:
    if total == 0:
        return 0.0
    z = _normal_quantile(1.0 - alpha)
    phat = successes / total
    denom = 1.0 + (z * z) / total
    center = phat + (z * z) / (2.0 * total)
    margin = z * sqrt((phat * (1.0 - phat) + (z * z) / (4.0 * total)) / total)
    return max(0.0, (center - margin) / denom)


def _normal_quantile(p: float) -> float:
    # Acklam approximation, deterministic and dependency-free.
    a = [-3.969683028665376e01, 2.209460984245205e02, -2.759285104469687e02,
         1.383577518672690e02, -3.066479806614716e01, 2.506628277459239e00]
    b = [-5.447609879822406e01, 1.615858368580409e02, -1.556989798598866e02,
         6.680131188771972e01, -1.328068155288572e01]
    c = [-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e00,
         -2.549732539343734e00, 4.374664141464968e00, 2.938163982698783e00]
    d = [7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e00,
         3.754408661907416e00]
    plow = 0.02425
    phigh = 1.0 - plow
    if p <= 0.0:
        return -8.0
    if p >= 1.0:
        return 8.0
    if p < plow:
        q = sqrt(-2.0 * np.log(p))
        return (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / \
            ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0)
    if phigh < p:
        q = sqrt(-2.0 * np.log(1.0 - p))
        return -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / \
            ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0)
    q = p - 0.5
    r = q * q
    return (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q / \
        (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1.0)


def select_threshold(
    confidence: np.ndarray,
    proxy_pred: np.ndarray,
    oracle_label: np.ndarray,
    recall_target: float,
    delta: float,
) -> ThresholdResult:
    order = np.argsort(-confidence)
    sorted_conf = confidence[order]
    sorted_proxy = proxy_pred[order]
    sorted_oracle = oracle_label[order]

    unique_thresholds = np.unique(sorted_conf)
    best: ThresholdResult | None = None
    alpha_per_test = delta / max(len(unique_thresholds), 1)

    total = len(sorted_conf)
    for t in unique_thresholds:
        proxy_mask = sorted_conf >= t
        final_pred = np.where(proxy_mask, sorted_proxy, sorted_oracle)
        positives = sorted_oracle == 1
        positive_count = int(np.sum(positives))
        tp = int(np.sum((final_pred == 1) & positives))
        recall_emp = tp / positive_count if positive_count > 0 else 1.0
        recall_lb = wilson_lower_bound(tp, positive_count, alpha_per_test)
        coverage = float(np.sum(proxy_mask)) / total
        if recall_lb >= recall_target:
            candidate = ThresholdResult(t, coverage, recall_emp, recall_lb)
            if best is None or candidate.coverage > best.coverage:
                best = candidate

    if best is None:
        return ThresholdResult(1.1, 0.0, 1.0, 1.0)
    return best


def load_dataset(path: Path, feature_columns: Iterable[str], label_column: str, oracle_score_column: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    df = pd.read_csv(path)
    x = df[list(feature_columns)].to_numpy(dtype=float)
    y = df[label_column].to_numpy(dtype=int)
    oracle_score = df[oracle_score_column].to_numpy(dtype=float)
    return x, y, oracle_score


def main() -> None:
    parser = argparse.ArgumentParser(description="Run BARGAIN-R with LR proxy and correct+smooth oracle.")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--features", type=str, required=True, help="Comma-separated feature columns")
    parser.add_argument("--label-col", type=str, required=True)
    parser.add_argument("--oracle-score-col", type=str, required=True)
    parser.add_argument("--recall-target", type=float, default=0.9)
    parser.add_argument("--delta", type=float, default=0.05)
    args = parser.parse_args()

    feature_columns = [col.strip() for col in args.features.split(",") if col.strip()]
    x, y, oracle_score = load_dataset(args.input, feature_columns, args.label_col, args.oracle_score_col)

    n = len(x)
    train_end = int(0.6 * n)
    guarantee_start = int(0.8 * n)

    x_train, y_train = x[:train_end], y[:train_end]
    x_guarantee = x[guarantee_start:]
    oracle_guarantee = oracle_score[guarantee_start:]

    config = BargainRConfig(recall_target=args.recall_target, delta=args.delta)
    proxy = LogisticRegressionProxy(config.learning_rate, config.epochs, config.l2, config.seed)
    proxy.fit(x_train, y_train)

    probs = proxy.predict_proba(x_guarantee)
    conf = lr_confidence(probs)
    proxy_pred = (probs >= 0.5).astype(int)

    oracle = CorrectSmoothOracle()
    oracle_label = oracle.predict(oracle_guarantee)

    result = select_threshold(conf, proxy_pred, oracle_label, config.recall_target, config.delta)

    print("selected_threshold", result.threshold)
    print("proxy_coverage", result.coverage)
    print("recall_empirical", result.recall_empirical)
    print("recall_lower_bound", result.recall_lower_bound)


if __name__ == "__main__":
    main()
