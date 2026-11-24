"""Evaluate single edits and both orders of paired edits on unseen combinations."""
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


def fmt_vals(vals):
    return ", ".join("true" if v is True else "false" if v is False else str(v) for v in vals)


def main():
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
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, tok = load_model(args, device)

    rng = random.Random(args.seed)
    samples = [json.loads(l) for l in open(os.path.join(args.data, "test.jsonl"), encoding="utf-8") if l.strip()]
    rng.shuffle(samples)
    samples = samples[:args.limit]

    agg = {"n": 0, "single_ok": 0, "single_n": 0,
           "fwd_ok": 0, "rev_ok": 0, "order_invariant": 0,
           "preserved_ok": 0, "recomputed_sum": 0}
    rows = []
    for s in samples:
        if len(s["edits"]) < 2 or "combo_forward" not in s["reference"]:
            continue
        fam, ent = s["structure_family"], s["entity"]
        lay = C.family_layout(fam)
        lay["family"] = fam
        qname = s["queries"][0]
        out_names = list(lay["queries"][qname])
        qtext = f"{qname}?"
        n_nodes = len(lay["names"])

        def build_ws():
            rec = C.sample_to_record(s, "original", rng)
            g = C.collate([rec], device)
            with torch.no_grad():
                return NeuralWorkspace.build(model, g, device, lay)

        def read(ws):
            with torch.no_grad():
                return read_query(model, ws, qtext, out_names, device)

        def truth_from(ref):
            return fmt_vals([ref["states"][f"{ent}.{n}"] for n in out_names])

        e0, e1 = s["edits"][0], s["edits"][1]
        t0 = e0["target"].split(".", 1)[1]
        t1 = e1["target"].split(".", 1)[1]
        v0, v1 = e0["value"], e1["value"]

        # 单编辑基线
        for i, e in enumerate(s["edits"][:2]):
            ws = build_ws()
            tn = e["target"].split(".", 1)[1]
            with torch.no_grad():
                neural_edit(model, ws, tn, e["value"], device)
            ans = read(ws)
            ok = first_tok(ans) == truth_from(s["reference"]["single"][i])
            agg["single_ok"] += int(ok); agg["single_n"] += 1

        # 组合正序：edit0 -> edit1
        ws = build_ws()
        with torch.no_grad():
            st0 = neural_edit(model, ws, t0, v0, device)
            st1 = neural_edit(model, ws, t1, v1, device)
        ans_fwd = read(ws)
        truth_combo = truth_from(s["reference"]["combo_forward"])
        ok_fwd = first_tok(ans_fwd) == truth_combo
        # 保持率：第二次编辑后，非受影响节点版本/张量不变（neural_edit 返回 preserved）
        preserved_ok = st1["preserved_nodes"] == n_nodes - 1 - st1["recomputed_nodes"]
        agg["fwd_ok"] += int(ok_fwd)
        agg["preserved_ok"] += int(preserved_ok)
        agg["recomputed_sum"] += st1["recomputed_nodes"]

        # 组合反序：edit1 -> edit0
        ws = build_ws()
        with torch.no_grad():
            neural_edit(model, ws, t1, v1, device)
            neural_edit(model, ws, t0, v0, device)
        ans_rev = read(ws)
        ok_rev = first_tok(ans_rev) == truth_from(s["reference"]["combo_reverse"])
        agg["rev_ok"] += int(ok_rev)
        agg["order_invariant"] += int(first_tok(ans_fwd) == first_tok(ans_rev))
        agg["n"] += 1
        rows.append({"base_id": s["base_id"], "truth_combo": truth_combo,
                     "fwd": ans_fwd.strip(), "rev": ans_rev.strip(),
                     "ok_fwd": ok_fwd, "ok_rev": ok_rev,
                     "order_inv": first_tok(ans_fwd) == first_tok(ans_rev),
                     "recomputed": st1["recomputed_nodes"], "preserved_ok": preserved_ok})
        if len(rows) % 50 == 0:
            print(f"progress: {len(rows)} samples", flush=True)

    n = max(agg["n"], 1)
    sn = max(agg["single_n"], 1)
    summary = {
        "experiment": "Paired edit generalization after single-edit training",
        "ckpt": args.ckpt, "data": args.data, "n_samples": agg["n"],
        "single_edit_accuracy": agg["single_ok"] / sn,
        "combo_forward_accuracy": agg["fwd_ok"] / n,
        "combo_reverse_accuracy": agg["rev_ok"] / n,
        "order_invariance": agg["order_invariant"] / n,
        "preservation_rate": agg["preserved_ok"] / n,
        "mean_recomputed_nodes": agg["recomputed_sum"] / n,
        "generalization_gap": agg["single_ok"] / sn - agg["fwd_ok"] / n,
    }
    with open(os.path.join(args.out, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    with open(os.path.join(args.out, "rows.jsonl"), "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
