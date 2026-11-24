"""Belief-R LLM 基线：完全照搬官方 inference_local.py 的 logprob 分类协议。

官方协议（HLTCHKUST/belief-revision, EMNLP 2024）：
- time_t: 2 前提  "Premise 1: p1\nPremise 2: p2\nConclusion: [LABEL]"
- time_t1: 3 前提 "Premise 1: p1\nPremise 2: p2\nPremise 3: p3\nConclusion: [LABEL]"
- 对候选结论 a/b/c 分别算整段 logprob，取最高者 → 零样本分类。

与 belief_graph.py 同 641 对、同 a/b/c 口径，公平可比。
指标：time_t_acc / BU_Acc / BM_Acc / BREU。
"""
from __future__ import annotations

import os
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["HF_DATASETS_OFFLINE"] = "1"
import argparse
import json

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from interlat.evaluation.belief_graph import load_belief


@torch.inference_mode()
def seq_logprob(model, tokenizer, text, device):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=1024).to(device)
    logits = model(**inputs).logits
    output_ids = inputs["input_ids"][:, 1:]
    lp = torch.gather(F.log_softmax(logits, dim=-1), 2, output_ids.unsqueeze(2)).squeeze(-1)
    lp[inputs["attention_mask"][:, :-1] == 0] = 0
    return float(lp.sum(dim=1).cpu()[0])


def build_prompt(ex, step):
    lines = ex["questions"].split("\n")
    premises = [l for l in lines if l.strip() and not l.startswith("What") and not l.startswith("(")]
    if step == "time_t":
        premises = premises[:2]
    else:
        premises = premises[:3]
    parts = [f"Premise {i+1}: {p}" for i, p in enumerate(premises)]
    return "\n".join(parts) + "\nConclusion: "


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if not tok.pad_token:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model, trust_remote_code=True, torch_dtype=torch.float16).to(device).eval()

    data = load_belief()
    by_id = {}
    for ex in data:
        by_id.setdefault(ex["dataset_id"], {})[ex["step"]] = ex
    pairs = [v for v in by_id.values() if "time_t" in v and "time_t1" in v]
    if args.limit:
        pairs = pairs[:args.limit]
    print(f"pairs: {len(pairs)} model: {args.model}", flush=True)

    bu_ok = bu_n = bm_ok = bm_n = t_ok = t_n = 0
    rows = []
    for i, pr in enumerate(pairs):
        tt, tt1 = pr["time_t"], pr["time_t1"]
        # time_t 初始推理
        p0 = build_prompt(tt, "time_t")
        cand0 = [tt["a"], tt["b"], tt["c"]]
        lp0 = [seq_logprob(model, tok, p0 + lab, device) for lab in cand0]
        a0 = chr(ord("a") + int(np.argmax(lp0)))
        # time_t1 修正
        p1 = build_prompt(tt1, "time_t1")
        cand1 = [tt1["a"], tt1["b"], tt1["c"]]
        lp1 = [seq_logprob(model, tok, p1 + lab, device) for lab in cand1]
        a1 = chr(ord("a") + int(np.argmax(lp1)))
        gt0, gt1 = tt["ground_truth"], tt1["ground_truth"]
        t_ok += int(a0 == gt0); t_n += 1
        if gt1 != gt0:
            bu_ok += int(a1 == gt1); bu_n += 1
        else:
            bm_ok += int(a1 == gt1); bm_n += 1
        rows.append({"id": tt["dataset_id"], "gt0": gt0, "a0": a0, "gt1": gt1, "a1": a1})
        if (i + 1) % 50 == 0:
            print(f"  {i+1}/{len(pairs)} t_acc={t_ok/t_n:.3f}", flush=True)

    bu = bu_ok / max(bu_n, 1); bm = bm_ok / max(bm_n, 1)
    summary = {"method": "llm_logprob_official", "model": args.model,
               "n_pairs": len(pairs), "time_t_acc": t_ok / max(t_n, 1),
               "BU_Acc": bu, "BU_n": bu_n, "BM_Acc": bm, "BM_n": bm_n,
               "BREU": (bu + bm) / 2}
    with open(os.path.join(args.out, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    with open(os.path.join(args.out, "rows.jsonl"), "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
