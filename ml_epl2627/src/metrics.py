"""
Proper scoring rules, calibration error, and paired bootstrap significance tests
for EPL 2026/27 1X2 probabilistic forecasting.
Outcomes indexed as: 0 = Home Win (H), 1 = Draw (D), 2 = Away Win (A).
"""

import numpy as np


def clip_probs(probs: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    p = np.asarray(probs, dtype=float)
    p = np.clip(p, eps, 1.0 - eps)
    return p / p.sum(axis=1, keepdims=True)


def rps_per_match(y_true: np.ndarray, probs: np.ndarray) -> np.ndarray:
    """Ranked Probability Score for ordered 3-class outcomes (H=0, D=1, A=2)."""
    p = clip_probs(probs)
    y_oh = np.eye(3)[y_true]
    cdf_p = np.cumsum(p, axis=1)[:, :2]
    cdf_y = np.cumsum(y_oh, axis=1)[:, :2]
    return 0.5 * np.sum((cdf_p - cdf_y) ** 2, axis=1)


def log_loss_per_match(y_true: np.ndarray, probs: np.ndarray) -> np.ndarray:
    p = clip_probs(probs)
    idx = np.arange(len(y_true))
    return -np.log(p[idx, y_true])


def brier_per_match(y_true: np.ndarray, probs: np.ndarray) -> np.ndarray:
    p = clip_probs(probs)
    y_oh = np.eye(3)[y_true]
    return np.sum((p - y_oh) ** 2, axis=1)


def ece_mc(y_true: np.ndarray, probs: np.ndarray, n_bins: int = 5) -> float:
    """Expected Calibration Error on predicted max class confidence."""
    p = clip_probs(probs)
    conf = np.max(p, axis=1)
    pred = np.argmax(p, axis=1)
    correct = (pred == y_true).astype(float)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n = len(y_true)
    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        mask = (conf >= lo) & (conf <= hi if i == n_bins - 1 else conf < hi)
        if mask.any():
            ece += (mask.sum() / n) * abs(correct[mask].mean() - conf[mask].mean())
    return float(ece)


def evaluate_all(y_true: np.ndarray, probs: np.ndarray) -> dict:
    p = clip_probs(probs)
    pred = np.argmax(p, axis=1)
    return {
        "n": int(len(y_true)),
        "rps": float(np.mean(rps_per_match(y_true, p))),
        "log_loss": float(np.mean(log_loss_per_match(y_true, p))),
        "brier": float(np.mean(brier_per_match(y_true, p))),
        "accuracy": float(np.mean(pred == y_true)),
        "ece": ece_mc(y_true, p, n_bins=5),
    }


def bootstrap_rps_diff(
    y_true: np.ndarray,
    probs_model: np.ndarray,
    probs_base: np.ndarray,
    n_boot: int = 2000,
    seed: int = 42,
) -> dict:
    """
    Paired bootstrap of (RPS_model - RPS_base). Negative means model beats baseline.
    """
    d = rps_per_match(y_true, probs_model) - rps_per_match(y_true, probs_base)
    rng = np.random.default_rng(seed)
    n = len(d)
    idx = rng.integers(0, n, size=(n_boot, n))
    boot_means = d[idx].mean(axis=1)
    return {
        "mean_diff": float(d.mean()),
        "ci_low": float(np.percentile(boot_means, 2.5)),
        "ci_high": float(np.percentile(boot_means, 97.5)),
        "p_better": float(np.mean(boot_means < 0)),
    }
