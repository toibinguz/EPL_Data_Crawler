"""Evaluation metrics for H/D/A probabilities: log loss, RPS, Brier, accuracy and paired bootstrap."""
import numpy as np

from .config import BOOTSTRAP_B, RESULT_ORDER, SEED


def onehot(y):
    y = np.asarray(y)
    return np.stack([(y == k) for k in RESULT_ORDER], axis=1).astype(float)


def ll_vec(P, y):
    """Per-match log loss."""
    return -np.log(np.clip((P * onehot(y)).sum(1), 1e-15, 1))


def rps_vec(P, y):
    """Per-match Ranked Probability Score (ordered H < D < A)."""
    cP = np.cumsum(P, 1)[:, :2]
    cY = np.cumsum(onehot(y), 1)[:, :2]
    return ((cP - cY) ** 2).sum(1) / 2


def evaluate(P, y, name=""):
    y = np.asarray(y)
    return {"model": name, "n": len(y),
            "log_loss": ll_vec(P, y).mean(),
            "rps": rps_vec(P, y).mean(),
            "brier": ((P - onehot(y)) ** 2).sum(1).mean(),
            "accuracy": (np.array(RESULT_ORDER)[P.argmax(1)] == y).mean()}


def paired_bootstrap(loss_base, loss_new, B=BOOTSTRAP_B, seed=SEED):
    """delta = loss_new - loss_base (negative = new model is better). Returns delta, CI95% low/high, P(delta >= 0)."""
    d = np.asarray(loss_new) - np.asarray(loss_base)
    rng = np.random.default_rng(seed)
    boots = np.empty(B)
    for b in range(0, B, 200):                        # in batches to save memory
        idx = rng.integers(0, len(d), size=(min(200, B - b), len(d)))
        boots[b:b + idx.shape[0]] = d[idx].mean(1)
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return d.mean(), lo, hi, (boots >= 0).mean()
