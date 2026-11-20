#!/usr/bin/env python3
"""Training curves and a before/after table.

  python plot.py --log runs/qwen05b-grpo/train_log.csv --summaries results/base_summary.json results/grpo_summary.json
Writes results/training_curves.png and results/summary.md.
"""
import argparse
import csv
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def smooth(y, k=10):
    y = np.asarray(y, dtype=float)
    if len(y) < k:
        return y
    c = np.cumsum(np.insert(y, 0, 0.0))
    return np.concatenate([np.full(k - 1, np.nan), (c[k:] - c[:-k]) / k])


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--log", required=True)
    p.add_argument("--summaries", nargs="*", default=[])
    p.add_argument("--out", default="results")
    a = p.parse_args()
    os.makedirs(a.out, exist_ok=True)
    rows = list(csv.DictReader(open(a.log)))
    step = [int(r["step"]) for r in rows]
    panels = [("accuracy", "Training accuracy", "#2E62B8"), ("tool_use_rate", "Answers using the calculator", "#D9822B"),
              ("reward", "Mean reward", "#3B7D4F"), ("kl", "KL to the starting model", "#6B4C9A")]
    fig, axes = plt.subplots(2, 2, figsize=(11, 6.5))
    for ax, (key, title, color) in zip(axes.flat, panels):
        y = [float(r[key]) for r in rows]
        ax.plot(step, y, color=color, alpha=0.25, lw=1)
        ax.plot(step, smooth(y), color=color, lw=2.2)
        ax.set_title(title)
        ax.set_xlabel("GRPO step")
        ax.grid(alpha=0.3)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    fig.suptitle("GRPO training (thin: per step, thick: 10-step moving average)")
    fig.tight_layout()
    fig.savefig(os.path.join(a.out, "training_curves.png"), dpi=150)
    print("saved", os.path.join(a.out, "training_curves.png"))

    if a.summaries:
        S = [json.load(open(f)) for f in a.summaries]
        cols = [("accuracy", "Accuracy", "%.1f%%", 100), ("tool_use_rate", "Uses calculator", "%.1f%%", 100),
                ("tool_calls_per_problem", "Calls / problem", "%.2f", 1), ("tool_error_rate", "Tool errors", "%.1f%%", 100),
                ("answer_format_rate", "Answer format", "%.1f%%", 100), ("completion_tokens", "Tokens", "%.0f", 1)]
        lines = ["| Model | Problems | " + " | ".join(c[1] for c in cols) + " |", "|---|---|" + "---|" * len(cols)]
        for s in S:
            lines.append("| %s | %d | " % (s["name"], s["problems"]) + " | ".join(f % (s[k] * m) for k, _, f, m in cols) + " |")
        open(os.path.join(a.out, "summary.md"), "w").write("\n".join(lines) + "\n")
        print("\n".join(lines))


if __name__ == "__main__":
    main()
