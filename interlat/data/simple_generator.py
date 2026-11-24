"""Generate shallow arithmetic graphs for the single-operation curriculum."""
from __future__ import annotations

import json
import random
from dataclasses import asdict
from typing import Any

from ..workspace.schema import ARG_ROLES, Node, Workspace
from ..workspace import reference
from ..workspace.edit import Edit, apply_edit

ENTITY_PREFIXES = ["alpha", "bravo", "cedar", "delta", "ember", "fjord"]


def _make_ws(ent: str, a: int, b: int, op: str = "add") -> Workspace:
    ws = Workspace()
    ws.add(Node(node_id=f"{ent}.a", type="count", operator="input", value=a,
                local_fact=f"a={a}"))
    ws.add(Node(node_id=f"{ent}.b", type="count", operator="input", value=b,
                local_fact=f"b={b}"))
    ws.add(Node(node_id=f"{ent}.c", type="count", operator=op,
                parents=(f"{ent}.a", f"{ent}.b"), arg_roles=ARG_ROLES[op]))
    ws.queries = {"result": (f"{ent}.c",)}
    return ws


def make_sample(rng: random.Random, base_id: str, op: str = "add",
                max_val: int = 9, bound: int = 99) -> dict[str, Any]:
    ent = rng.choice(ENTITY_PREFIXES) + str(rng.randint(10, 99))
    a = rng.randint(0, max_val)
    b = rng.randint(0, max_val)
    if op == "sub" and a < b:
        a, b = b, a
    ws = _make_ws(ent, a, b, op)
    vals = reference.evaluate(ws)
    for nid, v in vals.items():
        ws.nodes[nid].value = v
    if not reference.check_range(vals, bound=bound):
        return make_sample(rng, base_id, op, max_val, bound)

    task_text = f"{ent} calculation: a={a}, b={b}. Find a {'+' if op=='add' else '-' if op=='sub' else '*'} b."

    def run(edits_seq):
        w = _make_ws(ent, a, b, op)
        for nid, v in reference.evaluate(w).items():
            w.nodes[nid].value = v
        stats = []
        for e in edits_seq:
            stats.append(apply_edit(w, Edit(e["target"], e["op"], e["value"])))
        vmap = {nid: n.value for nid, n in w.nodes.items()}
        return {"states": vmap,
                "answers": {q: [vmap[n] for n in outs] for q, outs in w.queries.items()},
                "edit_stats": stats}

    # 两个单编辑（改 a / 改 b），保证结果仍在范围内
    ea_val = rng.randint(0, max_val)
    eb_val = rng.randint(0, max_val)
    edits = [
        {"target": f"{ent}.a", "op": "set", "value": ea_val, "kind": "single"},
        {"target": f"{ent}.b", "op": "set", "value": eb_val, "kind": "single"},
    ]
    single_runs = [run([e]) for e in edits]
    combo_fwd = run(edits)
    combo_rev = run(list(reversed(edits)))
    for ref in single_runs + [combo_fwd, combo_rev]:
        if not reference.check_range(ref["states"], bound=bound):
            return make_sample(rng, base_id, op, max_val, bound)

    return {
        "base_id": base_id,
        "structure_family": f"simple_{op}",
        "entity": ent,
        "task_text": task_text,
        "graph": {"nodes": [asdict(n) for n in ws.nodes.values()],
                  "queries": {q: list(o) for q, o in ws.queries.items()}},
        "initial_state": {"a": a, "b": b},
        "edits": edits,
        "queries": list(ws.queries),
        "reference": {"original": run([]), "single": single_runs,
                      "combo_forward": combo_fwd, "combo_reverse": combo_rev},
        "split_group": f"simple_{op}",
    }


def generate(n: int, seed: int, op: str = "add", max_val: int = 9,
             bound: int = 99) -> list[dict]:
    rng = random.Random(seed)
    return [make_sample(rng, f"task_{i:06d}", op, max_val, bound) for i in range(n)]


def split(samples: list[dict], seed: int, dev_ratio=0.2, test_ratio=0.2):
    rng = random.Random(seed)
    ids = [s["base_id"] for s in samples]
    rng.shuffle(ids)
    nd, nt = int(len(ids) * dev_ratio), int(len(ids) * test_ratio)
    dev, test = set(ids[:nd]), set(ids[nd:nd + nt])
    out = {"train": [], "dev": [], "test": []}
    for s in samples:
        if s["base_id"] in dev:
            out["dev"].append(s)
        elif s["base_id"] in test:
            out["test"].append(s)
        else:
            out["train"].append(s)
    return out


def dump_jsonl(samples, path):
    with open(path, "w", encoding="utf-8") as f:
        for s in samples:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")
