"""Paired comparison: 1.7B front-end + workspace versus 1.7B 4-shot CoT.

Reads only archived E8 rows; writes frontend_vs_cot4.json next to this file.
Bootstrap is stratified by trajectory length (3/4/5; 8 bookings per stratum),
seed 20261120, 10,000 replicates, matching paired_comparison.py.
"""
import json
import math
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent


def read_rows(path):
    return [json.loads(line) for line in path.read_text(encoding="utf8").splitlines()]


def load(mode):
    return {tuple(r["key"]): bool(r["correct"])
            for r in read_rows(HERE / "results" / mode / "rows.jsonl")}


def sequence_flags(data):
    seqs = {}
    for k, ok in data.items():
        seqs.setdefault(k[:2], []).append(ok)
    assert len(seqs) == 48
    return {k: all(v) for k, v in seqs.items()}


def mcnemar(a, b):
    keys = sorted(set(a) & set(b))
    b01 = sum(a[k] and not b[k] for k in keys)
    b10 = sum((not a[k]) and b[k] for k in keys)
    n = b01 + b10
    if n == 0:
        return b01, b10, 1.0
    tail = sum(math.comb(n, i) for i in range(min(b01, b10) + 1)) / 2 ** n
    return b01, b10, min(1.0, 2 * tail)


def main():
    front = load("frontend")
    cot4 = load("small_cot4")
    flags_front = sequence_flags(front)
    flags_cot4 = sequence_flags(cot4)

    lengths = []
    diffs = []
    for booking in range(24):
        keys = [k for k in sorted(front) if k[0] == booking]
        assert len(keys) % 2 == 0
        lengths.append(len(keys) // 2)
        step = sum(front[k] for k in keys) - sum(cot4[k] for k in keys)
        traj = sum(flags_front[(booking, d)] for d in (-1, 1)) - sum(flags_cot4[(booking, d)] for d in (-1, 1))
        diffs.append((step, traj))
    values = np.array(diffs)
    rng = np.random.default_rng(20261120)
    draws = np.zeros((10000, 2))
    for L in (3, 4, 5):
        idx = np.flatnonzero(np.array(lengths) == L)
        assert len(idx) == 8
        draws += values[rng.choice(idx, (10000, 8))].sum(axis=1)
    ci = np.quantile(100 * draws / np.array([192, 48]), [0.025, 0.975], axis=0)

    b, c, p = mcnemar(flags_front, flags_cot4)
    out = {
        "reference": "1.7B 4-shot CoT (small_cot4)",
        "candidate": "1.7B field front-end + workspace (frontend)",
        "n_trajectories": 48,
        "trajectory_all_correct": {"frontend": sum(flags_front.values()), "cot4": sum(flags_cot4.values())},
        "mcnemar": {"b_front_only": b, "c_cot4_only": c, "two_sided_exact_p": p},
        "paired_difference_pp": {
            "steps": float(100 * values[:, 0].sum() / 192),
            "trajectories": float(100 * values[:, 1].sum() / 48),
        },
        "bootstrap_95_ci_pp": {
            "steps": [float(ci[0, 0]), float(ci[1, 0])],
            "trajectories": [float(ci[0, 1]), float(ci[1, 1])],
        },
        "bootstrap": "stratified by trajectory length (3/4/5; 8 bookings per stratum), 10000 replicates, seed 20261120",
    }
    large = {}
    for r in read_rows(HERE / "posthoc_format" / "rows.jsonl"):
        if r["mode"] == "large_cot":
            large[tuple(r["key"])] = bool(r["correct"])
    flags_large = sequence_flags(large)
    b2, c2, p2 = mcnemar(flags_front, flags_large)
    out["frontend_vs_large_cot_format_audit"] = {
        "trajectory_all_correct": {"frontend": sum(flags_front.values()), "large": sum(flags_large.values())},
        "mcnemar": {"b_front_only": b2, "c_large_only": c2, "two_sided_exact_p": p2},
    }
    (HERE / "frontend_vs_cot4.json").write_text(json.dumps(out, indent=2), encoding="utf8")
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
