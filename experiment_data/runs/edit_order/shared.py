"""Frozen arithmetic model, trace utilities, and output helpers for the CPU checklist."""
import hashlib
import json
import os
import platform
import random
import sys
from collections import defaultdict
from pathlib import Path

os.environ["OPENBLAS_NUM_THREADS"] = "1"
import numpy as np

source = Path(__file__).resolve().parents[1] / "arithmetic_workspace"
sys.path.insert(0, str(source))
from iterate import build, Candidate, evaluate, truth

ranges = {"digits": (0, 9), "unseen_two_digit": (60, 99), "unseen_three_digit": (100, 999)}

def frozen_model():
    model = Candidate.__new__(Candidate)
    model.canonical = True
    with np.load(source / "inputs.npz") as raw:
        model.names = list(raw["names"])
        model.features = {f"s = {i}": x.astype(float) for i, x in enumerate(raw["z"][model.names.index("s")])}
    with np.load(source / "calibration_42.npz") as fit:
        model.mean, model.bias, model.weights, model.place, model.operators = (
            fit[key] for key in ("mean", "bias", "decoder", "place", "operators"))
    model.decoded = {}
    return model

def artifacts(folder, **kwargs):
    hashes = {p: hashlib.sha256((source / p).read_bytes()).hexdigest()
              for p in ("iterate.py", "inputs.npz", "formal_rows_42.jsonl", "calibration_42.npz")}
    save(folder, "config", dict(fit_seed=42, python=platform.python_version(),
         numpy=np.__version__, source_hashes=hashes, **kwargs))

def save(folder, name, obj):
    (folder / f"{name}.json").write_text(json.dumps(obj, indent=2, allow_nan=False), encoding="utf-8")

def record(f, obj):
    f.write(json.dumps(obj, separators=(",", ":"), allow_nan=False) + "\n")

def problems():
    # Each task occurs twice, once per original edit. Keep the shared row and verify uniqueness.
    with (source / "formal_rows_42.jsonl").open() as f:
        rows = [r for line in f if (r := json.loads(line))["target"] == 0]
    assert len(rows) == len({tuple(identity(r).values()) for r in rows}) == 5400
    return rows

def identity(r):
    return dict(range=r.get("range", "digits"), mixed=r["mixed"], width=r["width"], task=r["task"])

def design_sample(n, seed):
    groups = defaultdict(list)
    for r in problems():
        groups[tuple(identity(r).values())[:3]].append(r)
    rng = random.Random(seed)
    rows = []
    for i, group in enumerate(groups.values()):
        rows.extend(rng.sample(group, n // len(groups) + int(i < n % len(groups))))
    rng.shuffle(rows)
    return rows

def change(rng, old, bounds):
    low, high = bounds
    return low + (old - low + rng.randint(1, high-low)) % (high-low+1)

def rebuild(model, vals, graph):
    return build([model.encode(n, v) for n, v in zip(model.names, vals)], graph, model.execute)

def edit(model, state, original, graph, target, new, oracle=truth):
    vals = original.copy()
    vals[target] = new
    local = state.copy()
    local[target] = model.encode(model.names[target], new)
    dirty = {target}
    for j, (op,l,r) in enumerate(graph, len(vals)):
        if l in dirty or r in dirty:
            local[j] = model.execute(op, local[l], local[r])
            dirty.add(j)
    gold = build(vals, graph, oracle)[-1]
    assert int(gold) == gold
    pred = round(local[-1])
    full = rebuild(model, vals, graph)
    return local, vals, dict(target=target, new=new, truth=int(gold), pred=pred, rebuild=round(full[-1]),
        correct=pred==gold, preserved=all(state[j]==local[j] for j in range(len(state)) if j not in dirty),
        agreement=pred==round(full[-1]), all_state_agreement=bool(local==full), executed=len(dirty)-1)
