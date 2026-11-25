"""Figures, saved as PNG into the run folder (no interactive display)."""
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from .config import BOOTSTRAP_B, RESULT_ORDER  # noqa: E402
from .metrics import onehot  # noqa: E402


def _save(fig, out_dir, name):
    fig.tight_layout()
    if out_dir is not None:
        fig.savefig(os.path.join(out_dir, name), dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_hda_by_season(ha, out_dir):
    """Home / draw / away rate per season (home advantage, relevant to H5)."""
    fig, ax = plt.subplots(figsize=(10, 3.5))
    ha.plot(ax=ax, marker="o", title="H / D / A rate per season")
    ax.axvline(2019.5, ls="--", c="grey")
    ax.text(2019.6, ha.values.max(), "COVID (no crowds)", fontsize=8)
    ax.set_ylabel("rate")
    _save(fig, out_dir, "hda_rate_by_season.png")


def plot_hypotheses(hyp_res, out_dir):
    """Delta log loss with bootstrap CI95% for every tested hypothesis."""
    pl = hyp_res.dropna(subset=["delta"])
    if pl.empty:
        return
    fig, ax = plt.subplots(figsize=(8, 0.5 * len(pl) + 1))
    colors = ["tab:green" if h < 0 else ("tab:red" if l > 0 else "tab:grey") for l, h in zip(pl.CI_low, pl.CI_high)]
    for k, (dm, l, h, col) in enumerate(zip(pl["delta"], pl.CI_low, pl.CI_high, colors)):
        ax.errorbar(dm, k, xerr=[[dm - l], [h - dm]], fmt="o", color=col, capsize=4)
    ax.axvline(0, ls="--", c="k")
    ax.set_yticks(range(len(pl)))
    ax.set_yticklabels(pl["H"])
    ax.invert_yaxis()
    ax.set_xlabel(f"delta log loss (negative = better), bootstrap CI95% ({BOOTSTRAP_B:,} samples)")
    ax.set_title("Hypothesis tests H1-H9 (rolling-origin 2015/16-2023/24)")
    _save(fig, out_dir, "hypothesis_tests.png")


def plot_blend_weight(ws, blend_ll, w_best, out_dir):
    fig, ax = plt.subplots(figsize=(6, 3))
    ax.plot(ws, blend_ll, marker=".")
    ax.axvline(w_best, ls="--")
    ax.set_xlabel("w (LR weight)")
    ax.set_ylabel("Dev log loss")
    ax.set_title("Blend weight selection")
    _save(fig, out_dir, "blend_weight.png")


def plot_reliability(P, y, model_name, out_dir):
    """Probability calibration (reliability) curves on Test, one panel per outcome."""
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8))
    Y = onehot(y)
    for k, ax in enumerate(axes):
        p = P[:, k]
        bins = np.quantile(p, np.linspace(0, 1, 9))
        bins[-1] += 1e-9
        b = np.digitize(p, bins[1:-1])
        xs = [p[b == i].mean() for i in range(8) if (b == i).any()]
        ys = [Y[b == i, k].mean() for i in range(8) if (b == i).any()]
        ax.plot([0, 1], [0, 1], "k--", lw=1)
        ax.plot(xs, ys, "o-")
        ax.set_title(f"P({RESULT_ORDER[k]}) - {model_name}")
        ax.set_xlabel("predicted probability")
        ax.set_ylabel("observed frequency")
        ax.set_xlim(0, max(0.8, max(xs) + .05))
        ax.set_ylim(0, max(0.8, max(ys) + .05))
    _save(fig, out_dir, "reliability_test.png")


def plot_accuracy_by_train_start(acc_tab, season_tab, start_years, always_h_dev, always_h_test, out_dir):
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.6))
    labels = [f"Train from {r}" for r in acc_tab.train_from]
    colors = plt.cm.tab10.colors

    # (1) Dev and Test accuracy
    ax = axes[0]
    x = np.arange(2)
    w = 0.8 / len(acc_tab)
    for k, r in acc_tab.iterrows():
        vals = [r.acc_dev, r.acc_test]
        bars = ax.bar(x + (k - (len(acc_tab) - 1) / 2) * w, vals, w, label=labels[k], color=colors[k])
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v + 0.004, f"{v:.1%}", ha="center", fontsize=9)
    for xi, base in zip(x, [always_h_dev, always_h_test]):
        ax.hlines(base, xi - 0.45, xi + 0.45, colors="grey", linestyles="--")
    ax.text(1, always_h_test - 0.012, "baseline: always H", ha="center", fontsize=8, color="grey")
    ax.set_xticks(x)
    ax.set_xticklabels(["Dev 2022-24", "Test 2024-26"])
    ax.set_ylim(0.35, max(acc_tab[["acc_dev", "acc_test"]].max()) + 0.06)
    ax.set_ylabel("accuracy")
    ax.set_title("Accuracy by train start year")
    ax.legend(fontsize=8, loc="lower left")

    # (2) Test accuracy per season
    ax = axes[1]
    seasons = sorted(season_tab.season.unique())
    xs = np.arange(len(seasons))
    for k, sy in enumerate(start_years):
        g = season_tab[season_tab.start_year == sy].set_index("season").reindex(seasons)
        bars = ax.bar(xs + (k - (len(start_years) - 1) / 2) * w, g.accuracy, w, label=labels[k], color=colors[k])
        for b, v in zip(bars, g.accuracy):
            ax.text(b.get_x() + b.get_width() / 2, v + 0.004, f"{v:.1%}", ha="center", fontsize=9)
    ax.set_xticks(xs)
    ax.set_xticklabels(seasons)
    ax.set_ylim(0.35, season_tab.accuracy.max() + 0.06)
    ax.set_title("Test accuracy per season")
    ax.legend(fontsize=8, loc="lower left")

    # (3) Log loss (lower = better) for comparison
    ax = axes[2]
    for k, r in acc_tab.iterrows():
        vals = [r.logloss_dev, r.logloss_test]
        bars = ax.bar(x + (k - (len(acc_tab) - 1) / 2) * w, vals, w, label=labels[k], color=colors[k])
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v + 0.002, f"{v:.3f}", ha="center", fontsize=9)
    ax.set_xticks(x)
    ax.set_xticklabels(["Dev", "Test"])
    lo = acc_tab[["logloss_dev", "logloss_test"]].values.min()
    hi = acc_tab[["logloss_dev", "logloss_test"]].values.max()
    ax.set_ylim(lo - 0.03, hi + 0.02)
    ax.set_title("Log loss (lower = better)")
    ax.legend(fontsize=8, loc="lower left")
    _save(fig, out_dir, "accuracy_by_train_start.png")
