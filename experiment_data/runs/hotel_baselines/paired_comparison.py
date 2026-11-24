"""Paired descriptive intervals relative to the archived parser plus learned workspace."""
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent


def read_rows(path):
    return [json.loads(line) for line in path.read_text(encoding="utf8").splitlines()]


def main():
    reference = {
        (r["pair"], r["direction"], r["step"]): r
        for r in read_rows(HERE.parent/"hotel_replay"/"cpu_rows.jsonl")
        if r["method"] == "learned_local"
    }
    comparison = {}
    for mode in ("small_cot0", "small_cot4", "frontend", "large_direct", "large_cot", "large_cot_format_audit"):
        audited = mode.endswith("_format_audit")
        records = (r for r in read_rows(HERE/"posthoc_format"/"rows.jsonl") if r["mode"]=="large_cot") if audited else read_rows(HERE/"results"/mode/"rows.jsonl")
        test = {tuple(r["key"]): r for r in records}
        assert test.keys() == reference.keys()
        lengths = []
        differences = []
        for booking in range(24):
            sequences = [sorted(k for k in reference if k[:2] == (booking, d)) for d in (-1, 1)]
            assert len(sequences[0]) == len(sequences[1])
            lengths.append(len(sequences[0]))
            keys = sequences[0] + sequences[1]
            if audited:
                assert all(test[k]["correct"] == (reference[k]["truth"]==test[k]["extracted"]) for k in keys)
            else:
                assert all(reference[k]["truth"] == test[k]["truth"] for k in keys)
            def score(data):
                return np.array([sum(data[k]["correct"] for k in keys),
                                 sum(all(data[k]["correct"] for k in seq) for seq in sequences)])
            differences.append(score(test) - score(reference))
        values = np.array(differences)
        rng = np.random.default_rng(20261120)
        draws = np.zeros((10000, 2))
        for length in (3, 4, 5):
            indices = np.flatnonzero(np.array(lengths) == length)
            assert len(indices) == 8
            draws += values[rng.choice(indices, (10000, 8))].sum(axis=1)
        denominators = np.array([192, 48])
        comparison[mode] = {
            "step_and_trajectory_difference_pp": (100*values.sum(axis=0)/denominators).tolist(),
            "lower_upper_95_pp": np.quantile(100*draws/denominators, [0.025,0.975], axis=0).tolist()
        }
    report = dict(reference="archived learned_local with rule parser",
                  unit="paired base booking; 8 bookings resampled within each of 3 lengths",
                  replicates=10000, seed=20261120, results=comparison)
    (HERE/"paired_comparison.json").write_text(json.dumps(report,indent=2), encoding="utf8")


if __name__ == "__main__":
    main()
