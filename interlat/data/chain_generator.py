"""多图节点课程：5 节点链 a+b→c, c+d→e, e+f→g。

编辑 a 只影响 c/e/g（3/5 节点重算），b/d/f 保持——体现 C3 依赖驱动局部重执行。
值 0～9，结果 0～99。查询 g。
"""
from __future__ import annotations

import json
import random
from dataclasses import asdict
from typing import Any

from ..workspace.schema import ARG_ROLES, Node, Workspace
from ..workspace import reference
from ..workspace.edit import Edit, apply_edit

ENTITY_PREFIXES = ["alpha", "bravo", "cedar", "delta", "ember", "fjord"]
INPUTS = ["a", "b", "d", "f"]
DERIVED = [("c", "add", ("a", "b")), ("e", "add", ("c", "d")), ("g", "add", ("e", "f"))]


def _make_ws(ent: str, vals: dict) -> Workspace:
    ws = Workspace()
    for name in INPUTS:
        ws.add(Node(node_id=f"{ent}.{name}", type="count", operator="input",
                    value=vals[name], local_fact=f"{name}={vals[name]}"))
    for name, op, parents in DERIVED:
        ws.add(Node(node_id=f"{ent}.{name}", type="count", operator=op,
                    parents=tuple(f"{ent}.{p}" for p in parents),
                    arg_roles=ARG_ROLES[op]))
    ws.queries = {"result": (f"{ent}.g",)}
    return ws


def make_sample(rng: random.Random, base_id: str) -> dict[str, Any]:
    ent = rng.choice(ENTITY_PREFIXES) + str(rng.randint(10, 99))
    vals = {n: rng.randint(0, 9) for n in INPUTS}
    ws = _make_ws(ent, vals)
    vmap = reference.evaluate(ws)
    for nid, v in vmap.items():
        ws.nodes[nid].value = v
    if not reference.check_range(vmap, bound=99):
        return make_sample(rng, base_id)
    task_text = f"{ent} 链式计算：a={vals['a']}, b={vals['b']}, d={vals['d']}, f={vals['f']}。c=a+b, e=c+d, g=e+f。求 g。"

    def run(edits_seq):
        w = _make_ws(ent, vals)
        for nid, v in reference.evaluate(w).items():
            w.nodes[nid].value = v
        stats = []
        for e in edits_seq:
            stats.append(apply_edit(w, Edit(e["target"], e["op"], e["value"])))
        vm = {nid: n.value for nid, n in w.nodes.items()}
        return {"states": vm,
                "answers": {q: [vm[n] for n in outs] for q, outs in w.queries.items()},
                "edit_stats": stats}

    edits = [{"target": f"{ent}.{n}", "op": "set", "value": rng.randint(0, 9),
              "kind": "single"} for n in ("a", "b")]
    single_runs = [run([e]) for e in edits]
    combo_fwd = run(edits)
    combo_rev = run(list(reversed(edits)))
    for ref in single_runs + [combo_fwd, combo_rev]:
        if not reference.check_range(ref["states"], bound=99):
            return make_sample(rng, base_id)

    return {
        "base_id": base_id, "structure_family": "chain5", "entity": ent,
        "task_text": task_text,
        "graph": {"nodes": [asdict(n) for n in ws.nodes.values()],
                  "queries": {q: list(o) for q, o in ws.queries.items()}},
        "initial_state": vals, "edits": edits, "queries": list(ws.queries),
        "reference": {"original": run([]), "single": single_runs,
                      "combo_forward": combo_fwd, "combo_reverse": combo_rev},
        "split_group": "chain5",
    }


def generate(n: int, seed: int) -> list[dict]:
    rng = random.Random(seed)
    return [make_sample(rng, f"task_{i:06d}") for i in range(n)]


def split(samples, seed, dev_ratio=0.2, test_ratio=0.2):
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
