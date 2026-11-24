"""Verify generated targets, composition, dependency updates and preservation."""
from __future__ import annotations

import argparse
import json
import os
import time
from collections import Counter

from ..data import generator
from ..workspace.schema import Node, Workspace
from ..workspace import reference
from ..workspace.edit import Edit, apply_edit


def rebuild(sample: dict, overrides: dict[str, int]) -> dict:
    """Rebuild the graph from its edited inputs."""
    inputs = dict(sample["initial_state"])
    inputs.update(overrides)
    ws = generator.build_workspace(__import__("random").Random(0),
                                   sample["structure_family"], inputs,
                                   sample["entity"])
    vals = reference.evaluate(ws)
    return vals


def overrides_from_edits(sample: dict, edits: list[dict]) -> dict[str, int]:
    out: dict[str, int] = {}
    for e in edits:
        name = e["target"].split(".", 1)[1]
        cur = out.get(name, sample["initial_state"][name])
        if e["op"] == "set":
            out[name] = e["value"]
        elif e["op"] == "add":
            out[name] = cur + e["value"]
        else:
            out[name] = int(not cur)
    return out


def validate_sample(sample: dict) -> list[str]:
    errs: list[str] = []
    fam, ent = sample["structure_family"], sample["entity"]
    inputs = sample["initial_state"]

    # 1 原版一致
    ws = generator.build_workspace(__import__("random").Random(0), fam, inputs, ent)
    vals = reference.evaluate(ws)
    if vals != sample["reference"]["original"]["states"]:
        errs.append("original states mismatch")
    if not reference.check_range(vals):
        errs.append("original out of range")

    # 2/4/7 单编辑：路线B vs 路线A，保持率，重算数
    for i, e in enumerate(sample["edits"][:2]):
        ws = generator.build_workspace(__import__("random").Random(0), fam, inputs, ent)
        for nid, v in reference.evaluate(ws).items():
            ws.nodes[nid].value = v
        before = {nid: (n.value, n.version) for nid, n in ws.nodes.items()}
        affected = ws.descendants({e["target"]})
        stats = apply_edit(ws, Edit(e["target"], e["op"], e["value"]))
        after = {nid: n.value for nid, n in ws.nodes.items()}
        route_a = rebuild(sample, overrides_from_edits(sample, [e]))
        if after != route_a:
            errs.append(f"single edit {i}: route B != route A")
        if after != sample["reference"]["single"][i]["states"]:
            errs.append(f"single edit {i}: stored reference mismatch")
        if not reference.check_range(after):
            errs.append(f"single edit {i}: out of range")
        for nid, (v, ver) in before.items():
            if nid != e["target"] and nid not in affected:
                if ws.nodes[nid].value != v or ws.nodes[nid].version != ver:
                    errs.append(f"single edit {i}: unrelated node {nid} changed")
        if stats["recomputed_nodes"] != len(affected):
            errs.append(f"single edit {i}: recomputed count mismatch")

    # 3 组合顺序交换一致
    fwd = sample["reference"]["combo_forward"]["states"]
    rev = sample["reference"]["combo_reverse"]["states"]
    if fwd != rev:
        errs.append("independent combo not commutative")
    ws = generator.build_workspace(__import__("random").Random(0), fam, inputs, ent)
    for nid, v in reference.evaluate(ws).items():
        ws.nodes[nid].value = v
    for e in sample["edits"][:2]:
        apply_edit(ws, Edit(e["target"], e["op"], e["value"]))
    if {nid: n.value for nid, n in ws.nodes.items()} != fwd:
        errs.append("combo sequential != stored reference")
    route_a = rebuild(sample, overrides_from_edits(sample, sample["edits"][:2]))
    if fwd != route_a:
        errs.append("combo route B != route A")
    if not reference.check_range(fwd):
        errs.append("combo out of range")
    return errs


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=20261114)
    ap.add_argument("--out", type=str, required=True)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    t0 = time.time()
    samples = generator.generate(args.n, args.seed)
    gen_time = time.time() - t0

    t1 = time.time()
    failures: dict[str, list[str]] = {}
    for s in samples:
        errs = validate_sample(s)
        if errs:
            failures[s["base_id"]] = errs
    val_time = time.time() - t1

    splits = generator.split_by_family(samples, args.seed + 1)
    id_sets = {k: {s["base_id"] for s in v} for k, v in splits.items()}
    leak = (id_sets["train"] & id_sets["dev"]) | (id_sets["train"] & id_sets["test"]) \
        | (id_sets["dev"] & id_sets["test"])

    fam_counts = Counter(s["structure_family"] for s in samples)
    node_counts = [len(s["graph"]["nodes"]) for s in samples]
    recomputed = [st["recomputed_nodes"]
                  for s in samples for st in s["reference"]["single"][0]["edit_stats"]]

    summary = {
        "experiment": "Task generator and symbolic reference validation",
        "n_samples": args.n,
        "seed": args.seed,
        "generation_seconds": round(gen_time, 3),
        "validation_seconds": round(val_time, 3),
        "samples_per_second": round(args.n / gen_time, 1),
        "failed_samples": len(failures),
        "failure_details": dict(list(failures.items())[:20]),
        "split_sizes": {k: len(v) for k, v in splits.items()},
        "split_leak_ids": sorted(leak),
        "family_counts": dict(fam_counts),
        "nodes_per_graph": {"min": min(node_counts), "max": max(node_counts),
                            "mean": round(sum(node_counts) / len(node_counts), 2)},
        "single_edit_recomputed_nodes": {
            "min": min(recomputed), "max": max(recomputed),
            "mean": round(sum(recomputed) / len(recomputed), 2)},
        "all_checks_passed": len(failures) == 0 and len(leak) == 0,
    }
    with open(os.path.join(args.out, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    for name, part in splits.items():
        generator.dump_jsonl(part, os.path.join(args.out, f"{name}.jsonl"))
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
