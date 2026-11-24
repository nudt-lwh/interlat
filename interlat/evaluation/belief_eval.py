"""Belief-R 信念修正评测：把我们的可编辑工作空间对准 AGM/Belief-R 战场。

Belief-R 结构（ΔR 框架）：
- time_t：前提集 → 初始信念（多选 a/b/c）
- time_t1：加入新前提 → 修正后信念（多选）
指标：BU-Acc（该更新时更新）、BM-Acc（不该更新时保持）、BREU=平均。

本评测先用骨干 LLM 的 CoT 基线跑 Belief-R，建立对照基准；
我们的方法（结构化信念状态＋编辑）作为对照接入。
"""
from __future__ import annotations

import os
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
import argparse
import json
import re

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def load_belief():
    from datasets import load_dataset
    d = load_dataset("CAiRE/belief_r")
    return d["test"]


def build_prompt(ex, cot=True):
    q = ex["questions"]
    if cot:
        return (q + "\nLet's reason step by step about what necessarily follows, "
                "then give the final answer as (a), (b), or (c).\nAnswer:")
    return q + "\nThe answer is ("


def extract_answer(text):
    # 取最后出现的 (a)/(b)/(c) 或 a/b/c
    m = re.findall(r"\(([abc])\)", text.lower())
    if m:
        return m[-1]
    m = re.findall(r"\b([abc])\b", text.lower().strip()[-20:])
    return m[-1] if m else ""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="experiment_data/artifacts/Qwen3-1.7B")
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=400)
    ap.add_argument("--no-cot", action="store_true")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    data = load_belief()
    # 按 dataset_id 配对 time_t / time_t1
    by_id = {}
    for ex in data:
        by_id.setdefault(ex["dataset_id"], {})[ex["step"]] = ex
    pairs = [v for v in by_id.values() if "time_t" in v and "time_t1" in v]
    pairs = pairs[:args.limit]
    print(f"pairs: {len(pairs)}", flush=True)

    tok = AutoTokenizer.from_pretrained(args.model)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16).to(device)
    model.eval()

    @torch.no_grad()
    def answer(ex):
        p = build_prompt(ex, cot=not args.no_cot)
        enc = tok([p], return_tensors="pt", truncation=True, max_length=512).to(device)
        gen = model.generate(**enc, max_new_tokens=200 if not args.no_cot else 8,
                             do_sample=False, pad_token_id=tok.pad_token_id)
        txt = tok.decode(gen[0][enc["input_ids"].shape[1]:], skip_special_tokens=True)
        return extract_answer(txt), txt

    bu_ok = bu_n = bm_ok = bm_n = 0
    t_ok = t_n = 0
    rows = []
    for pr in pairs:
        tt, tt1 = pr["time_t"], pr["time_t1"]
        a0, _ = answer(tt)
        a1, txt1 = answer(tt1)
        gt0, gt1 = tt["ground_truth"], tt1["ground_truth"]
        t_ok += int(a0 == gt0); t_n += 1
        # 该更新：gt1 != gt0；不该更新：gt1 == gt0
        if gt1 != gt0:
            bu_ok += int(a1 == gt1); bu_n += 1
        else:
            bm_ok += int(a1 == gt1); bm_n += 1
        rows.append({"id": tt["dataset_id"], "gt0": gt0, "a0": a0,
                     "gt1": gt1, "a1": a1, "modus": tt1["modus"]})
        if len(rows) % 50 == 0:
            print(f"progress {len(rows)}", flush=True)

    bu = bu_ok / max(bu_n, 1)
    bm = bm_ok / max(bm_n, 1)
    summary = {
        "model": args.model, "cot": not args.no_cot, "n_pairs": len(pairs),
        "time_t_acc": t_ok / max(t_n, 1),
        "BU_Acc": bu, "BU_n": bu_n,
        "BM_Acc": bm, "BM_n": bm_n,
        "BREU": (bu + bm) / 2,
    }
    with open(os.path.join(args.out, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    with open(os.path.join(args.out, "rows.jsonl"), "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
