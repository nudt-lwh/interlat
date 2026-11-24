"""符号结构化状态＋增量执行 强对照（方案 v1.0 必需对照）。

与神经版共享同一图与输入。比较：质量（精确）、状态存储字节、
首次构建/编辑/重建墙钟、组合泛化。符号法天然精确，本对照用于
诚实界定潜在表示的净价值（存储/耗时/组合），不宣称质量领先。

用法：python -m interlat.evaluation.eval_symbolic --data <dir> --out <dir> --limit 300
"""
from __future__ import annotations

import argparse
import json
import os
import random
import time

from ..workspace.schema import Workspace, Node, ARG_ROLES
from ..workspace import reference, edit as sym_edit


def build_ws(sample) -> Workspace:
    ws = Workspace()
    for nd in sample["graph"]["nodes"]:
        op = nd["operator"]
        roles = tuple(nd.get("arg_roles") or ARG_ROLES.get(op, ()))
        ws.add(Node(node_id=nd["node_id"], type=nd["type"], operator=op,
                    parents=tuple(nd["parents"]), arg_roles=roles,
                    value=nd["value"], local_fact=nd.get("local_fact", "")))
    for q, outs in sample["graph"]["queries"].items():
        ws.queries[q] = tuple(outs)
    return ws


def read_vals(ws, sample, q):
    return [ws.nodes[n].value for n in sample["graph"]["queries"][q]]


def fmt(vals):
    return ", ".join("true" if v is True else "false" if v is False else str(v) for v in vals)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=300)
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    rng = random.Random(args.seed)
    samples = [json.loads(l) for l in open(os.path.join(args.data, "test.jsonl"), encoding="utf-8") if l.strip()]
    rng.shuffle(samples)
    samples = samples[:args.limit]

    agg = {"n": 0, "orig_ok": 0, "single_ok": 0, "single_n": 0,
           "combo_ok": 0, "preserved_ok": 0,
           "build_ms": [], "edit_ms": [], "rebuild_ms": [],
           "sym_bytes": 0, "nodes": 0}
    for s in samples:
        q = s["queries"][0]
        # 首次构建
        t0 = time.perf_counter()
        ws = build_ws(s)
        reference.evaluate(ws)
        build_ms = (time.perf_counter() - t0) * 1000
        agg["build_ms"].append(build_ms)
        agg["sym_bytes"] += sum(28 for _ in ws.nodes)  # python int/bool 对象近似
        agg["nodes"] += len(ws.nodes)
        agg["orig_ok"] += int(fmt(read_vals(ws, s, q)) == fmt(s["reference"]["original"]["answers"][q]))

        # 单编辑（增量）vs 真值
        for i, e in enumerate(s["edits"][:2]):
            ws = build_ws(s); reference.evaluate(ws)
            t0 = time.perf_counter()
            st = sym_edit.apply_edit(ws, sym_edit.Edit(target=e["target"], op="set", value=e["value"]))
            edit_ms = (time.perf_counter() - t0) * 1000
            agg["edit_ms"].append(edit_ms)
            ok = fmt(read_vals(ws, s, q)) == fmt(s["reference"]["single"][i]["answers"][q])
            agg["single_ok"] += int(ok); agg["single_n"] += 1
            agg["preserved_ok"] += int(st["preserved_nodes"] == len(ws.nodes) - 1 - st["recomputed_nodes"])
            # 完整重建对照
            s_mod = json.loads(json.dumps(s))
            tgt = e["target"].split(".", 1)[1]
            s_mod["initial_state"][tgt] = e["value"]
            for nd in s_mod["graph"]["nodes"]:
                if nd["node_id"] == e["target"]:
                    nd["value"] = e["value"]
            t0 = time.perf_counter()
            ws2 = build_ws(s_mod); reference.evaluate(ws2)
            rebuild_ms = (time.perf_counter() - t0) * 1000
            agg["rebuild_ms"].append(rebuild_ms)

        # 组合（正序）
        if len(s["edits"]) >= 2 and "combo_forward" in s["reference"]:
            ws = build_ws(s); reference.evaluate(ws)
            for e in s["edits"][:2]:
                sym_edit.apply_edit(ws, sym_edit.Edit(target=e["target"], op="set", value=e["value"]))
            agg["combo_ok"] += int(fmt(read_vals(ws, s, q)) == fmt(s["reference"]["combo_forward"]["answers"][q]))
        agg["n"] += 1

    n = max(agg["n"], 1)
    summary = {
        "method": "symbolic_structured_incremental",
        "data": args.data, "n": agg["n"],
        "original_accuracy": agg["orig_ok"] / n,
        "single_edit_accuracy": agg["single_ok"] / max(agg["single_n"], 1),
        "combo_accuracy": agg["combo_ok"] / n,
        "preservation_rate": agg["preserved_ok"] / max(agg["single_n"], 1),
        "mean_build_ms": sum(agg["build_ms"]) / n,
        "mean_edit_ms": sum(agg["edit_ms"]) / max(len(agg["edit_ms"]), 1),
        "mean_rebuild_ms": sum(agg["rebuild_ms"]) / max(len(agg["rebuild_ms"]), 1),
        "speedup": sum(agg["rebuild_ms"]) / max(sum(agg["edit_ms"]), 1e-9),
        "mean_state_bytes": agg["sym_bytes"] / n,
        "mean_nodes": agg["nodes"] / n,
    }
    with open(os.path.join(args.out, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
