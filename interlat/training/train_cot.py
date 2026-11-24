"""显式 CoT 对照训练（实验03/04 的公平对照）。

同一骨干 Qwen3-1.7B、同一 LoRA 配置、同一数据划分；输入为任务文本，
目标为符号参考器生成的逐步计算轨迹＋最终答案（教师轨迹来自可执行真值，
与 InterLat 的状态监督同源，保证公平）。

用法：
  python -m interlat.training.train_cot --data experiment_data/runs/exp03_formal_data \
      --model experiment_data/artifacts/Qwen3-1.7B --out experiment_data/runs/exp03_cot_5090 \
      --epochs 2 --micro-batch 16 --grad-accum 4
"""
from __future__ import annotations

import argparse
import json
import os
import time

import torch
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer

from ..data.generator import FAMILIES


def cot_text(sample: dict, variant: str) -> tuple[str, str]:
    """从参考状态重建逐步计算轨迹（算子语义→自然语言算式）。"""
    fam, ent = sample["structure_family"], sample["entity"]
    spec = FAMILIES[fam]
    ref = sample["reference"]
    if variant == "original":
        states, answers = ref["original"]["states"], ref["original"]["answers"]
        prefix = sample["task_text"]
    elif variant.startswith("single"):
        i = int(variant[-1])
        states, answers = ref["single"][i]["states"], ref["single"][i]["answers"]
        e = sample["edits"][i]
        name = e["target"].split(".", 1)[1]
        prefix = sample["task_text"] + f" 条件更新：{name} 改为 {e['value']}。"
    else:
        states, answers = ref["combo_forward"]["states"], ref["combo_forward"]["answers"]
        parts = []
        for e in sample["edits"][:2]:
            name = e["target"].split(".", 1)[1]
            parts.append(f"{name} 改为 {e['value']}")
        prefix = sample["task_text"] + " 条件更新：" + "，".join(parts) + "。"

    lines = []
    sym = {"add": "+", "sub": "-", "mul": "*", "floordiv": "//",
           "gt": ">", "lt": "<", "min": "min", "max": "max"}
    for name, op, parents in spec["derived"]:
        vals = [states[f"{ent}.{p}"] for p in parents]
        out = states[f"{ent}.{name}"]
        if op in ("gt", "lt"):
            lines.append(f"{vals[0]} {sym[op]} {vals[1]} = {out}")
        elif op in ("min", "max"):
            lines.append(f"{sym[op]}({vals[0]}, {vals[1]}) = {out}")
        else:
            lines.append(f"{vals[0]} {sym[op]} {vals[1]} = {out}")
    qname = list(spec["queries"])[0]
    ans = answers[qname]
    ans_str = ", ".join("true" if v is True else "false" if v is False else str(v)
                        for v in ans)
    target = "\n".join(lines) + f"\n答案: {ans_str}"
    return prefix + "\n请逐步计算并给出答案。\n", target


def save_trainable(model, path: str, step: int) -> None:
    sd = {k: v for k, v in model.state_dict().items() if "lora_" in k}
    torch.save({"model": sd, "step": step}, path)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--variants", default="original")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--micro-batch", type=int, default=16)
    ap.add_argument("--grad-accum", type=int, default=4)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--lora-r", type=int, default=16)
    ap.add_argument("--eval-every", type=int, default=500)
    ap.add_argument("--max-steps", type=int, default=None)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, "config.json"), "w", encoding="utf-8") as f:
        json.dump(vars(args), f, ensure_ascii=False, indent=2)
    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    variants = tuple(args.variants.split(","))

    def load(split):
        recs = []
        with open(os.path.join(args.data, f"{split}.jsonl"), encoding="utf-8") as f:
            for line in f:
                s = json.loads(line)
                for v in variants:
                    p, t = cot_text(s, v)
                    recs.append((p, t))
                if args.limit and len(recs) >= args.limit:
                    break
        return recs[:args.limit] if args.limit else recs

    train_recs, dev_recs = load("train"), load("dev")[:200]
    print(f"train {len(train_recs)}, dev {len(dev_recs)}", flush=True)

    tok = AutoTokenizer.from_pretrained(args.model, padding_side="left")
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16)
    cfg = LoraConfig(r=args.lora_r, lora_alpha=2 * args.lora_r, lora_dropout=0.05,
                     target_modules=["q_proj", "v_proj"], task_type="CAUSAL_LM")
    model = get_peft_model(model, cfg)
    model.to(device)
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.01)

    def make_batch(recs):
        prompts = [p for p, _ in recs]
        targets = [t for _, t in recs]
        full = [p + t + tok.eos_token for p, t in recs]
        enc = tok(full, return_tensors="pt", padding=True, truncation=True,
                  max_length=384).to(device)
        plen = tok(prompts, return_tensors="pt", padding=True, truncation=True,
                   max_length=384)["attention_mask"].sum(1)
        labels = enc["input_ids"].clone()
        for i in range(labels.shape[0]):
            labels[i, :plen[i]] = -100
        labels[enc["attention_mask"] == 0] = -100
        return enc, labels

    @torch.no_grad()
    def evaluate(max_batches=10):
        model.eval()
        correct = tot = 0
        for i in range(0, min(len(dev_recs), max_batches * args.micro_batch),
                       args.micro_batch):
            recs = dev_recs[i:i + args.micro_batch]
            prompts = [p for p, _ in recs]
            enc = tok(prompts, return_tensors="pt", padding=True, truncation=True,
                      max_length=256).to(device)
            gen = model.generate(**enc, max_new_tokens=160, do_sample=False,
                                 pad_token_id=tok.pad_token_id)
            texts = tok.batch_decode(gen[:, enc["input_ids"].shape[1]:],
                                     skip_special_tokens=True)
            for txt, (_, t) in zip(texts, recs):
                tot += 1
                truth = t.split("答案: ")[-1].strip()
                if truth in txt:
                    correct += 1
        model.train()
        return {"dev_answer_acc": correct / max(tot, 1), "dev_n": tot}

    import random
    rng = random.Random(args.seed)
    log, step = [], 0
    t_start = time.time()
    model.train()
    for ep in range(args.epochs):
        idx = list(range(len(train_recs)))
        rng.shuffle(idx)
        for i in range(0, len(idx), args.micro_batch):
            recs = [train_recs[j] for j in idx[i:i + args.micro_batch]]
            enc, labels = make_batch(recs)
            loss = model(**enc, labels=labels).loss
            (loss / args.grad_accum).backward()
            if (step + 1) % args.grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(params, 1.0)
                opt.step()
                opt.zero_grad()
            step += 1
            if step % 20 == 0:
                el = time.time() - t_start
                rec = {"epoch": ep, "step": step, "loss": float(loss.detach()),
                       "seconds": round(el, 1),
                       "samples_per_s": round(step * args.micro_batch / el, 2)}
                if device == "cuda":
                    rec["vram_peak_gb"] = round(torch.cuda.max_memory_allocated() / 2**30, 2)
                log.append(rec)
                print(json.dumps(rec, ensure_ascii=False), flush=True)
            if step % args.eval_every == 0:
                ev = evaluate()
                ev["step"] = step
                log.append({"eval": ev})
                print("EVAL " + json.dumps(ev, ensure_ascii=False), flush=True)
                save_trainable(model, os.path.join(args.out, f"ckpt_step{step}.pt"), step)
            if args.max_steps and step >= args.max_steps:
                break
        if args.max_steps and step >= args.max_steps:
            break

    ev = evaluate(max_batches=25)
    save_trainable(model, os.path.join(args.out, "ckpt_final.pt"), step)
    summary = {"experiment": args.out, "final_eval": ev, "total_steps": step,
               "total_seconds": round(time.time() - t_start, 1),
               "train_records": len(train_recs)}
    with open(os.path.join(args.out, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    with open(os.path.join(args.out, "train_log.jsonl"), "w", encoding="utf-8") as f:
        for rec in log:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
