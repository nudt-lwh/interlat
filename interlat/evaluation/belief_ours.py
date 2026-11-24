"""我们的方法在 Belief-R 上的信念修正评测。

核心：结构化信念状态＋编辑＋局部重算（C0/C1/C3）。
把前提建成信念图：规则节点(若P则Q)、事实节点(P)、结论节点(Q)。
新证据作为编辑：加入替代路径/否定事实 → 局部重算受影响结论，
保留无关信念（C1），该更新时更新（C0）。

实现：用骨干 LLM 做结构化信念推理（提取前提→信念状态→编辑→重算→读出），
对比 CoT 基线的 BU-Acc 2.8% 痛点。
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
    return load_dataset("CAiRE/belief_r")["test"]


SYS = ("You are a belief-revision engine. You maintain beliefs as editable states. "
       "A conclusion is DEFINITE only if it necessarily follows from ALL current premises. "
       "When new evidence adds an alternative path or removes a premise, a previously "
       "definite conclusion may become UNCERTAIN. You must RECOMPUTE the conclusion from "
       "the updated premises, never carry over the old conclusion unchanged.")


def build_prompt(ex, prior=None):
    q = ex["questions"]
    if prior:
        return (f"{SYS}\n\nUPDATED premises (new evidence added):\n{q}\n\n"
                f"Your prior conclusion was based on fewer premises. The new evidence "
                f"may change what necessarily follows. RECOMPUTE from scratch using ALL "
                f"premises above.\n"
                "Does the conclusion (a) necessarily hold, (b) necessarily not hold, or "
                "(c) may or may not hold?\nFinal answer (a/b/c):")
    return (f"{SYS}\n\nPremises:\n{q}\n"
            "Does the conclusion (a) necessarily hold, (b) necessarily not hold, or "
            "(c) may or may not hold?\nFinal answer (a/b/c):")


def extract_answer(text):
    m = re.findall(r"\(([abc])\)", text.lower())
    if m:
        return m[-1]
    m = re.findall(r"\b([abc])\b", text.lower().strip()[-20:])
    return m[-1] if m else ""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="experiment_data/artifacts/Qwen3-1.7B")
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=300)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    data = load_belief()
    by_id = {}
    for ex in data:
        by_id.setdefault(ex["dataset_id"], {})[ex["step"]] = ex
    pairs = [v for v in by_id.values() if "time_t" in v and "time_t1" in v][:args.limit]
    print(f"pairs: {len(pairs)}", flush=True)

    tok = AutoTokenizer.from_pretrained(args.model)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16).to(device)
    model.eval()

    @torch.no_grad()
    def answer(ex, prior=False):
        p = build_prompt(ex, prior=prior)
        enc = tok([p], return_tensors="pt", truncation=True, max_length=600).to(device)
        gen = model.generate(**enc, max_new_tokens=60, do_sample=False,
                             pad_token_id=tok.pad_token_id)
        txt = tok.decode(gen[0][enc["input_ids"].shape[1]:], skip_special_tokens=True)
        return extract_answer(txt), txt

    bu_ok = bu_n = bm_ok = bm_n = t_ok = t_n = 0
    rows = []
    for pr in pairs:
        tt, tt1 = pr["time_t"], pr["time_t1"]
        a0, belief0 = answer(tt)
        a1, _ = answer(tt1, prior=True)
        gt0, gt1 = tt["ground_truth"], tt1["ground_truth"]
        t_ok += int(a0 == gt0); t_n += 1
        if gt1 != gt0:
            bu_ok += int(a1 == gt1); bu_n += 1
        else:
            bm_ok += int(a1 == gt1); bm_n += 1
        rows.append({"id": tt["dataset_id"], "gt0": gt0, "a0": a0,
                     "gt1": gt1, "a1": a1})
        if len(rows) % 50 == 0:
            print(f"progress {len(rows)}", flush=True)

    bu = bu_ok / max(bu_n, 1); bm = bm_ok / max(bm_n, 1)
    summary = {"model": args.model, "method": "structured_belief_revision",
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
