"""P0-5 多查询复用（实验07）：建一次工作空间答多查询 vs 每问重建。

条件A：build 一次，依次读 q1..qk（复用节点状态；reader 每次新建 KV， unless --reader-cache）。
条件B：每个查询各自重建后读。
指标：每查询准确率、累计墙钟 ms、speedup=B/A。复杂族准确率低时仍报时间，准确率仅作参考。

用法：python -m interlat.evaluation.eval_multiquery --model <骨干> --ckpt <ckpt> \
   --data experiment_data/runs/exp03_formal_data --out <dir> --limit 200
"""
from __future__ import annotations

import argparse
import json
import os
import random
import time

import torch
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer

from ..models.workspace_model import InterLatModel
from ..models.neural_edit import NeuralWorkspace, read_query, ReaderKVCache
from ..training import collate as C


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


def first_tok(t):
    return t.strip().split()[0] if t.strip() else ""


def fmt(vals):
    return ", ".join("true" if v is True else "false" if v is False else str(v) for v in vals)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=200)
    ap.add_argument("--lora-r", type=int, default=16)
    ap.add_argument("--node-dim", type=int, default=256)
    ap.add_argument("--node-tokens", type=int, default=2)
    ap.add_argument("--exec-layers", type=int, default=4)
    ap.add_argument("--reader-cache", action="store_true")
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, tok = load_model(args, device)
    rng = random.Random(args.seed)
    samples = [json.loads(l) for l in open(os.path.join(args.data, "test.jsonl"), encoding="utf-8") if l.strip()]
    rng.shuffle(samples)
    samples = [s for s in samples if len(s["queries"]) >= 2][:args.limit]

    cache = ReaderKVCache() if args.reader_cache else None
    agg = {"n": 0, "A_ms": [], "B_ms": [], "A_ok": 0, "B_ok": 0, "q": 0}
    for s in samples:
        fam, ent = s["structure_family"], s["entity"]
        lay = C.family_layout(fam); lay["family"] = fam
        rec = C.sample_to_record(s, "original", rng)
        g = C.collate([rec], device)
        queries = s["queries"][:2]
        ref = s["reference"]["original"]["answers"]

        def read(ws, q):
            out_names = list(lay["queries"][q])
            if cache is not None:
                return cache.read(model, ws, f"{q}?", out_names, device)
            return read_query(model, ws, f"{q}?", out_names, device)

        # 条件A：建一次，读两个查询
        t0 = time.perf_counter()
        with torch.no_grad():
            ws = NeuralWorkspace.build(model, g, device, lay)
            a_ans = [read(ws, q) for q in queries]
        A_ms = (time.perf_counter() - t0) * 1000
        # 条件B：每问重建
        t0 = time.perf_counter()
        b_ans = []
        with torch.no_grad():
            for q in queries:
                ws_b = NeuralWorkspace.build(model, g, device, lay)
                b_ans.append(read(ws_b, q))
        B_ms = (time.perf_counter() - t0) * 1000

        for i, q in enumerate(queries):
            truth = fmt(ref[q])
            agg["A_ok"] += int(first_tok(a_ans[i]) == truth)
            agg["B_ok"] += int(first_tok(b_ans[i]) == truth)
            agg["q"] += 1
        agg["A_ms"].append(A_ms); agg["B_ms"].append(B_ms)
        agg["n"] += 1
        if agg["n"] % 50 == 0:
            print(f"progress {agg['n']}", flush=True)

    n = max(agg["n"], 1)
    summary = {
        "experiment": "P0-5 multi-query reuse", "ckpt": args.ckpt, "data": args.data,
        "n_tasks": agg["n"], "n_queries": agg["q"], "reader_cache": bool(args.reader_cache),
        "A_build_once_ms": sum(agg["A_ms"]) / n, "B_rebuild_each_ms": sum(agg["B_ms"]) / n,
        "speedup_B_over_A": sum(agg["B_ms"]) / max(sum(agg["A_ms"]), 1e-9),
        "A_acc": agg["A_ok"] / max(agg["q"], 1), "B_acc": agg["B_ok"] / max(agg["q"], 1),
    }
    with open(os.path.join(args.out, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
