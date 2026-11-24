"""P1-2 鲁棒性审计：实体/变量重命名、输入顺序、模板改写（不改图、不改值）。

对照原始 test，报告各扰动下准确率/路线一致率/保持率的下降。
训练未做这些增广，预期可能显著下降（诚实捷径风险）。

用法：python -m interlat.evaluation.eval_robust --model <骨干> --ckpt <ckpt> \
   --data <dir> --out <dir> --limit 200
"""
from __future__ import annotations

import argparse
import json
import os
import random

import torch
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer

from ..models.workspace_model import InterLatModel
from ..models.neural_edit import NeuralWorkspace, neural_edit, read_query
from ..training import collate as C

RENAME = {"a": "x1", "b": "x2", "d": "y1", "f": "y2"}


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


def transform(texts, mode, rng):
    if mode == "rename":
        out = []
        for t in texts:
            name, val = t.split(" = ")
            out.append(f"{RENAME.get(name, name)} = {val}")
        return out
    if mode == "order":
        idx = list(range(len(texts))); rng.shuffle(idx)
        return [texts[i] for i in idx]
    if mode == "template":
        return [f"the value of {t.split(' = ')[0]} is {t.split(' = ')[1]}" for t in texts]
    return texts


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
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, tok = load_model(args, device)
    rng = random.Random(args.seed)
    samples = [json.loads(l) for l in open(os.path.join(args.data, "test.jsonl"), encoding="utf-8") if l.strip()]
    rng.shuffle(samples)
    samples = samples[:args.limit]

    modes = ["original", "rename", "order", "template"]
    agg = {m: {"ok": 0, "n": 0, "agree": 0, "preserved": 0} for m in modes}
    for s in samples:
        fam, ent = s["structure_family"], s["entity"]
        lay = C.family_layout(fam); lay["family"] = fam
        q = s["queries"][0]
        out_names = list(lay["queries"][q])
        e = s["edits"][0]
        tgt = e["target"].split(".", 1)[1]
        truth = fmt(s["reference"]["single"][0]["answers"][q])
        rec = C.sample_to_record(s, "original", rng)
        for mode in modes:
            g = C.collate([rec], device)
            g.input_texts = [transform(rec["input_texts"], mode, rng)]
            with torch.no_grad():
                ws = NeuralWorkspace.build(model, g, device, lay)
                st = neural_edit(model, ws, tgt, e["value"], device)
                ans_b = read_query(model, ws, f"{q}?", out_names, device)
                # rebuild route for agreement
                s_mod = json.loads(json.dumps(s))
                s_mod["initial_state"][tgt] = e["value"]
                s_mod["reference"]["original"] = s["reference"]["single"][0]
                rec_a = C.sample_to_record(s_mod, "original", rng)
                g_a = C.collate([rec_a], device)
                g_a.input_texts = [transform(rec_a["input_texts"], mode, rng)]
                ws_a = NeuralWorkspace.build(model, g_a, device, lay)
                ans_a = read_query(model, ws_a, f"{q}?", out_names, device)
            ok = first_tok(ans_b) == truth
            agg[mode]["ok"] += int(ok); agg[mode]["n"] += 1
            agg[mode]["agree"] += int(first_tok(ans_a) == first_tok(ans_b))
            agg[mode]["preserved"] += int(st["preserved_nodes"] == len(lay["names"]) - 1 - st["recomputed_nodes"])
        if agg["original"]["n"] % 50 == 0:
            print(f"progress {agg['original']['n']}", flush=True)

    summary = {"experiment": "P1-2 robustness audit", "ckpt": args.ckpt, "data": args.data,
               "by_mode": {m: {"accuracy": agg[m]["ok"] / max(agg[m]["n"], 1),
                               "route_agreement": agg[m]["agree"] / max(agg[m]["n"], 1),
                               "preservation": agg[m]["preserved"] / max(agg[m]["n"], 1),
                               "n": agg[m]["n"]} for m in modes}}
    with open(os.path.join(args.out, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
