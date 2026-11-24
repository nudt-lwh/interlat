"""Inspect state-head predictions and generated answers."""
from __future__ import annotations

import argparse
import json
import os

import torch
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer

from ..models.workspace_model import InterLatModel, MAX_DIGITS
from ..training import collate as C


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=200)
    ap.add_argument("--lora-r", type=int, default=16)
    ap.add_argument("--node-dim", type=int, default=256)
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    tok = AutoTokenizer.from_pretrained(args.model)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    base = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16)
    cfg = LoraConfig(r=args.lora_r, lora_alpha=2 * args.lora_r, lora_dropout=0.05,
                     target_modules=["q_proj", "v_proj"], task_type="CAUSAL_LM")
    base = get_peft_model(base, cfg, adapter_name="encoder")
    base.add_adapter("reader", cfg)
    base.to(device)
    model = InterLatModel(base, tok, node_dim=args.node_dim)
    model.to(device)
    for m in (model.enc_proj, model.read_proj, model.executor, model.state_head):
        m.to(torch.bfloat16)
    ck = torch.load(args.ckpt, map_location=device, weights_only=True)
    model.load_state_dict(ck["model"], strict=False)
    model.eval()

    import random
    rng = random.Random(args.seed)
    samples = []
    with open(os.path.join(args.data, "dev.jsonl"), encoding="utf-8") as f:
        for line in f:
            samples.append(json.loads(line))
    rng.shuffle(samples)
    samples = samples[:args.limit]

    state_correct = state_total = 0
    input_correct = input_total = 0
    derived_correct = derived_total = 0
    ans_correct = ans_total = 0
    examples = []
    with torch.no_grad():
        for s in samples:
            rec = C.sample_to_record(s, "original", rng)
            g = C.collate([rec], device)
            z_in = model.encode_inputs(g.input_texts, device)
            z_all = model.execute_all(z_in, g)
            labels = g.state_labels.to(device)
            sign_logits, digit_logits, bool_logits = model.state_head(
                z_all.reshape(-1, 2, model.node_dim))
            sign_pred = sign_logits.argmax(-1)
            dig_pred = digit_logits.argmax(-1)
            bool_pred = bool_logits.argmax(-1)
            lay = C.family_layout(s["structure_family"])
            n_in = len(lay["names"]) - len(lay["derived"])
            for j in range(labels.shape[1]):
                is_bool_node = bool((labels[0, j, 2] == 10) and (labels[0, j, 1] <= 1))
                if is_bool_node:
                    ok = bool(bool_pred[j] == labels[0, j, 1])
                else:
                    ok = bool((sign_pred[j] == labels[0, j, 0]) and
                              (dig_pred[j] == labels[0, j, 1:]).all())
                state_correct += int(ok)
                state_total += 1
                if j < n_in:
                    input_correct += int(ok)
                    input_total += 1
                else:
                    derived_correct += int(ok)
                    derived_total += 1
            ids = model.read_answers(z_all, g, device, generate=True, max_new_tokens=8)
            txt = tok.batch_decode(ids, skip_special_tokens=True)[0]
            first = txt.strip().split()[0] if txt.strip() else ""
            ok = first == rec["answer_text"].strip()
            ans_correct += int(ok)
            ans_total += 1
            if len(examples) < 40:
                examples.append({"query": rec["query_text"],
                                 "truth": rec["answer_text"],
                                 "generated": txt.strip(), "ok": ok,
                                 "family": s["structure_family"]})

    summary = {
        "ckpt": args.ckpt, "n_samples": len(samples),
        "state_readout_acc": state_correct / max(state_total, 1),
        "input_state_acc": input_correct / max(input_total, 1),
        "derived_state_acc": derived_correct / max(derived_total, 1),
        "answer_acc": ans_correct / max(ans_total, 1),
        "examples": examples,
    }
    with open(os.path.join(args.out, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(json.dumps({k: v for k, v in summary.items() if k != "examples"},
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
