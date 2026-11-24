"""实验04：单编辑主效果评估（神经工作空间）。

对每个测试样本：
- 路线A：从编辑后输入完整重建工作空间并读取答案；
- 路线B：原工作空间上神经编辑＋局部重执行后读取答案；
指标：两路线答案准确率（对真值）、路线间一致率、无关节点保持率、
重算节点数、编辑耗时 vs 重建耗时。

用法：
  python -m interlat.evaluation.eval_edit --model <骨干> --ckpt <ckpt_final.pt> \
      --data experiment_data/runs/exp03_formal_data --out experiment_data/runs/exp04_eval \
      --limit 300
"""
from __future__ import annotations

import argparse
import json
import os
import time

import torch
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer

from ..models.workspace_model import InterLatModel
from ..models.neural_edit import NeuralWorkspace, neural_edit, read_query, ReaderKVCache
from ..training import collate as C
from ..data.generator import FAMILIES


def load_model(args, device):
    tok = AutoTokenizer.from_pretrained(args.model)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    base = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16)
    cfg = LoraConfig(r=args.lora_r, lora_alpha=2 * args.lora_r, lora_dropout=0.05,
                     target_modules=["q_proj", "v_proj"], task_type="CAUSAL_LM")
    base = get_peft_model(base, cfg, adapter_name="encoder")
    base.add_adapter("reader", cfg)
    base.to(device)
    model = InterLatModel(base, tok, node_dim=args.node_dim,
                          node_tokens=args.node_tokens, exec_layers=args.exec_layers)
    model.to(device)
    for m in (model.enc_proj, model.read_proj, model.executor, model.state_head):
        m.to(torch.bfloat16)
    ck = torch.load(args.ckpt, map_location=device, weights_only=True)
    model.load_state_dict(ck["model"], strict=False)
    model.eval()
    return model, tok


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=300)
    ap.add_argument("--lora-r", type=int, default=16)
    ap.add_argument("--node-dim", type=int, default=256)
    ap.add_argument("--node-tokens", type=int, default=2)
    ap.add_argument("--exec-layers", type=int, default=4)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--reader-cache", action="store_true",
                    help="P0-3：查询KV复用＋状态前缀按版本失效（默认关）")
    ap.add_argument("--breakdown", action="store_true",
                    help="P0-4：encode/execute/read 分项计时，写 summary_breakdown.json")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, tok = load_model(args, device)

    import random
    rng = random.Random(args.seed)
    samples = []
    with open(os.path.join(args.data, "test.jsonl"), encoding="utf-8") as f:
        for line in f:
            samples.append(json.loads(line))
    rng.shuffle(samples)
    samples = samples[:args.limit]

    rows = []
    agg = {"n": 0, "route_a_correct": 0, "route_b_correct": 0, "agree": 0,
           "preserved_ok": 0, "recomputed_sum": 0,
           "global_edit_correct": 0, "global_preserved_ok": 0,
           "edit_ms": [], "rebuild_ms": []}
    cache = ReaderKVCache() if args.reader_cache else None
    bd = {"edit_encode_ms": [], "edit_execute_ms": [], "edit_read_ms": [],
          "rebuild_encode_ms": [], "rebuild_execute_ms": [], "rebuild_read_ms": []}

    def do_read(ws, qtext, out_names):
        if cache is not None:
            return cache.read(model, ws, qtext, out_names, device)
        return read_query(model, ws, qtext, out_names, device)

    def manual_build(g, lay):
        """分项计时的重建：encode 与 execute 分开计时。"""
        t0 = time.perf_counter()
        z_in = model.encode_inputs(g.input_texts, device)
        enc = (time.perf_counter() - t0) * 1000
        t0 = time.perf_counter()
        z_all = model.execute_all(z_in, g)
        exe = (time.perf_counter() - t0) * 1000
        N = z_all.shape[1]
        n_in = len(lay["names"]) - len(lay["derived"])
        parents = [()] * n_in; roles = [()] * n_in; ops = [0] * n_in
        for (op, ps, rs) in lay["derived"]:
            parents.append(tuple(ps)); roles.append(tuple(rs)); ops.append(op)
        ws = NeuralWorkspace(z=z_all, versions=[0] * N, topo=list(range(N)),
                             parents=parents, roles=roles, ops=ops,
                             input_mask=[i < n_in for i in range(N)],
                             family=lay.get("family", ""), entity="", names=lay["names"])
        return ws, enc, exe

    def manual_edit(ws, target_short, new_val):
        idx = ws.names.index(target_short)
        t0 = time.perf_counter()
        z_new = model.encode_inputs([[f"{target_short} = {new_val}"]], device)[0, 0]
        enc = (time.perf_counter() - t0) * 1000
        ws.z[0, idx] = z_new
        ws.versions[idx] += 1
        affected = ws.descendants(idx)
        t0 = time.perf_counter()
        ex = model.executor
        rec = 0
        for j in ws.topo:
            if j not in affected:
                continue
            pz = ws.z[0, list(ws.parents[j])].unsqueeze(0)
            roles = torch.tensor([ws.roles[j]], device=device)
            x = ex.embed_parents(pz, roles)
            ops_t = torch.tensor([ws.ops[j]], device=device)
            ws.z[0, j] = ex.forward_single(x, ops_t)[0]
            ws.versions[j] += 1
            rec += 1
        exe = (time.perf_counter() - t0) * 1000
        preserved = sum(1 for i in range(len(ws.topo))
                        if i != idx and i not in affected)
        return {"recomputed_nodes": rec, "preserved_nodes": preserved,
                "affected": sorted(affected)}, enc, exe
    for s in samples:
        fam, ent = s["structure_family"], s["entity"]
        lay = C.family_layout(fam)
        lay["family"] = fam
        rec_orig = C.sample_to_record(s, "original", rng)
        g = C.collate([rec_orig], device)
        with torch.no_grad():
            ws = NeuralWorkspace.build(model, g, device, lay)
        # 每个样本评两个单编辑
        for i, e in enumerate(s["edits"][:2]):
            target_short = e["target"].split(".", 1)[1]
            new_val = e["value"] if e["op"] == "set" else \
                s["initial_state"][target_short] + e["value"]
            ref = s["reference"]["single"][i]
            qname = rng.choice(list(lay["queries"]))
            out_names = list(lay["queries"][qname])
            truth_vals = [ref["states"][f"{ent}.{n}"] for n in out_names]
            truth = ", ".join("true" if v is True else "false" if v is False
                              else str(v) for v in truth_vals)
            qtext = f"{qname}?"

            # 路线B：编辑＋局部重执行
            t0 = time.time()
            with torch.no_grad():
                if args.breakdown:
                    stats, e_enc, e_exe = manual_edit(ws, target_short, new_val)
                else:
                    stats = neural_edit(model, ws, target_short, new_val, device)
                    e_enc = e_exe = None
                tr0 = time.perf_counter()
                ans_b = do_read(ws, qtext, out_names)
                e_read = (time.perf_counter() - tr0) * 1000
            edit_ms = (time.time() - t0) * 1000
            if args.breakdown:
                bd["edit_encode_ms"].append(e_enc); bd["edit_execute_ms"].append(e_exe)
                bd["edit_read_ms"].append(e_read)

            # 路线A：完整重建（编辑后输入），参考真值用编辑后状态
            s_mod = json.loads(json.dumps(s))
            s_mod["initial_state"][target_short] = new_val
            s_mod["reference"]["original"] = ref
            rec_a = C.sample_to_record(s_mod, "original", rng)
            g_a = C.collate([rec_a], device)
            t0 = time.time()
            with torch.no_grad():
                if args.breakdown:
                    ws_a, r_enc, r_exe = manual_build(g_a, lay)
                else:
                    ws_a = NeuralWorkspace.build(model, g_a, device, lay)
                    r_enc = r_exe = None
                tr0 = time.perf_counter()
                ans_a = do_read(ws_a, qtext, out_names)
                r_read = (time.perf_counter() - tr0) * 1000
            rebuild_ms = (time.time() - t0) * 1000
            if args.breakdown:
                bd["rebuild_encode_ms"].append(r_enc); bd["rebuild_execute_ms"].append(r_exe)
                bd["rebuild_read_ms"].append(r_read)

            # 消融：全局向量编辑（所有输入节点重编码＋全量重执行，无局部定位）
            with torch.no_grad():
                ws_g = NeuralWorkspace.build(model, g, device, lay)
                new_texts = []
                for nm in lay["names"][:sum(ws_g.input_mask)]:
                    v = new_val if nm == target_short else s["initial_state"][nm]
                    new_texts.append(f"{nm} = {v}")
                z_g = model.encode_inputs([new_texts], device)
                ws_g.z[0, :z_g.shape[1]] = z_g[0]
                # 全量重算派生节点
                for j in ws_g.topo:
                    if ws_g.input_mask[j]:
                        continue
                    ps = ws_g.parents[j]
                    pz = ws_g.z[0, list(ps)].unsqueeze(0)
                    roles = torch.tensor([ws_g.roles[j]], device=device)
                    x = model.executor.embed_parents(pz, roles)
                    ops_t = torch.tensor([ws_g.ops[j]], device=device)
                    ws_g.z[0, j] = model.executor.forward_single(x, ops_t)[0]
                ans_g = read_query(model, ws_g, qtext, out_names, device)
            ok_g = (ans_g.strip().split()[0] if ans_g.strip() else "") == truth
            agg["global_edit_correct"] += int(ok_g)

            def first_tok(t):
                return t.strip().split()[0] if t.strip() else ""
            ok_a = first_tok(ans_a) == truth
            ok_b = first_tok(ans_b) == truth
            # 保持率：非受影响节点版本与张量不变（neural_edit 已校验）
            n_nodes = len(lay["names"])
            preserved_ok = stats["preserved_nodes"] == n_nodes - 1 - stats["recomputed_nodes"]
            agg["n"] += 1
            agg["route_a_correct"] += int(ok_a)
            agg["route_b_correct"] += int(ok_b)
            agg["agree"] += int(first_tok(ans_a) == first_tok(ans_b))
            agg["preserved_ok"] += int(preserved_ok)
            agg["recomputed_sum"] += stats["recomputed_nodes"]
            agg["edit_ms"].append(edit_ms)
            agg["rebuild_ms"].append(rebuild_ms)
            rows.append({"base_id": s["base_id"], "edit": i, "target": target_short,
                         "truth": truth, "route_a": ans_a.strip(), "route_b": ans_b.strip(),
                         "ok_a": ok_a, "ok_b": ok_b,
                         "recomputed": stats["recomputed_nodes"],
                         "edit_ms": round(edit_ms, 1), "rebuild_ms": round(rebuild_ms, 1)})
            # 恢复工作空间供下一个编辑：重建原版
            with torch.no_grad():
                if args.breakdown:
                    ws, _, _ = manual_build(g, lay)
                else:
                    ws = NeuralWorkspace.build(model, g, device, lay)
        if len(rows) % 50 == 0:
            print(f"progress: {len(rows)} edits", flush=True)

    n = max(agg["n"], 1)
    summary = {
        "experiment": "实验04：单编辑主效果（神经）",
        "ckpt": args.ckpt, "n_edits": agg["n"],
        "route_a_accuracy": agg["route_a_correct"] / n,
        "route_b_accuracy": agg["route_b_correct"] / n,
        "route_agreement": agg["agree"] / n,
        "global_edit_accuracy": agg["global_edit_correct"] / n,
        "preservation_rate": agg["preserved_ok"] / n,
        "mean_recomputed_nodes": agg["recomputed_sum"] / n,
        "mean_edit_ms": sum(agg["edit_ms"]) / n,
        "mean_rebuild_ms": sum(agg["rebuild_ms"]) / n,
        "speedup": (sum(agg["rebuild_ms"]) / max(sum(agg["edit_ms"]), 1e-9)),
    }
    with open(os.path.join(args.out, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    with open(os.path.join(args.out, "rows.jsonl"), "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    if args.breakdown:
        def mean(x):
            return sum(x) / max(len(x), 1)
        breakdown = {k: mean(v) for k, v in bd.items()}
        breakdown["reader_cache"] = bool(args.reader_cache)
        breakdown["edit_execute_vs_rebuild"] = (
            breakdown["rebuild_execute_ms"] / max(breakdown["edit_execute_ms"], 1e-9))
        breakdown["read_share_of_edit"] = (
            breakdown["edit_read_ms"] / max(breakdown["edit_encode_ms"] +
                                            breakdown["edit_execute_ms"] +
                                            breakdown["edit_read_ms"], 1e-9))
        with open(os.path.join(args.out, "summary_breakdown.json"), "w", encoding="utf-8") as f:
            json.dump(breakdown, f, ensure_ascii=False, indent=2)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
