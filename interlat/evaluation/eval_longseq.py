"""Evaluate sequences of edits against the symbolic reference."""
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
from ..workspace.schema import Workspace, Node, ARG_ROLES
from ..workspace import reference, edit as sym_edit


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


def build_sym(sample):
    ws = Workspace()
    for nd in sample["graph"]["nodes"]:
        op = nd["operator"]
        roles = tuple(nd.get("arg_roles") or ARG_ROLES.get(op, ()))
        ws.add(Node(node_id=nd["node_id"], type=nd["type"], operator=op,
                    parents=tuple(nd["parents"]), arg_roles=roles, value=nd["value"]))
    for q, outs in sample["graph"]["queries"].items():
        ws.queries[q] = tuple(outs)
    reference.evaluate(ws)
    return ws


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
    ap.add_argument("--limit", type=int, default=150)
    ap.add_argument("--lengths", default="3,4,5")
    ap.add_argument("--lora-r", type=int, default=16)
    ap.add_argument("--node-dim", type=int, default=256)
    ap.add_argument("--node-tokens", type=int, default=2)
    ap.add_argument("--exec-layers", type=int, default=4)
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, tok = load_model(args, device)
    lengths = [int(x) for x in args.lengths.split(",")]

    rng = random.Random(args.seed)
    vrng = random.Random(args.seed + 1)
    samples = [json.loads(l) for l in open(os.path.join(args.data, "test.jsonl"), encoding="utf-8") if l.strip()]
    rng.shuffle(samples)
    samples = samples[:args.limit]

    agg = {L: {"ok": 0, "n": 0, "preserved_ok": 0} for L in lengths}
    rows = []
    for s in samples:
        fam, ent = s["structure_family"], s["entity"]
        lay = C.family_layout(fam); lay["family"] = fam
        q = s["queries"][0]
        out_names = list(lay["queries"][q])
        input_names = [n for n in lay["names"][:lay["n_in"]]]
        n_nodes = len(lay["names"])
        rec = C.sample_to_record(s, "original", rng)
        g = C.collate([rec], device)

        for L in lengths:
            # 生成 L 个编辑（随机输入节点、随机新值 0..9）
            edits = [(vrng.choice(input_names), vrng.randint(0, 9)) for _ in range(L)]
            # 符号真值
            sym = build_sym(s)
            for tn, v in edits:
                sym_edit.apply_edit(sym, sym_edit.Edit(target=f"{ent}.{tn}", op="set", value=v))
            truth = fmt([sym.nodes[f"{ent}.{n}"].value for n in out_names])
            # 神经序列编辑
            with torch.no_grad():
                ws = NeuralWorkspace.build(model, g, device, lay)
                last = None
                for tn, v in edits:
                    last = neural_edit(model, ws, tn, v, device)
                ans = read_query(model, ws, f"{q}?", out_names, device)
            ok = first_tok(ans) == truth
            preserved = last["preserved_nodes"] == n_nodes - 1 - last["recomputed_nodes"]
            agg[L]["ok"] += int(ok); agg[L]["n"] += 1
            agg[L]["preserved_ok"] += int(preserved)
            rows.append({"base_id": s["base_id"], "length": L, "truth": truth,
                         "ans": ans.strip(), "ok": ok, "preserved": preserved})
        if len(rows) % 100 == 0:
            print(f"progress {len(rows)}", flush=True)

    summary = {"experiment": "long-sequence edits (trained single-edit only)",
               "ckpt": args.ckpt, "data": args.data,
               "by_length": {str(L): {"accuracy": agg[L]["ok"] / max(agg[L]["n"], 1),
                                      "preservation": agg[L]["preserved_ok"] / max(agg[L]["n"], 1),
                                      "n": agg[L]["n"]} for L in lengths}}
    with open(os.path.join(args.out, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    with open(os.path.join(args.out, "rows.jsonl"), "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
