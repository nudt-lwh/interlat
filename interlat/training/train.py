"""InterLat 训练脚本（实验03/04）。

用法（本地冒烟）：
  python -m interlat.training.train --data experiment_data/runs/exp02_generator_validation \
      --model experiment_data/artifacts/Qwen3-1.7B --out experiment_data/runs/exp03_smoke \
      --variants original --limit 400 --epochs 1 --micro-batch 4

云正式训练加 --grad-accum 8 --epochs 3 等；checkpoint 与指标存 out 目录。
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
from . import collate as C


def save_trainable(model, path: str, step: int) -> None:
    """只保存可训练参数（LoRA＋投影＋执行器＋状态头），避免3.4GB骨干重复落盘。"""
    sd = {k: v for k, v in model.state_dict().items()
          if ("lora_" in k) or k.startswith(("enc_proj", "read_proj", "executor",
                                             "state_head", "exec_proj", "exec_out"))}
    torch.save({"model": sd, "step": step}, path)


def load_trainable(model, path: str, device) -> int:
    ck = torch.load(path, map_location=device, weights_only=True)
    missing, unexpected = model.load_state_dict(ck["model"], strict=False)
    bad = [k for k in unexpected]
    if bad:
        raise ValueError(f"unexpected keys in checkpoint: {bad[:5]}")
    return ck.get("step", 0)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="含 train.jsonl/dev.jsonl 的目录")
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--variants", default="original",
                    help="逗号分隔：original,single0,single1,combo")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--micro-batch", type=int, default=4)
    ap.add_argument("--grad-accum", type=int, default=4)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--lambda-s", type=float, default=0.5)
    ap.add_argument("--lambda-aux", type=float, default=0.0,
                    help="状态读出→答案辅助路径权重（强监督）")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--lora-r", type=int, default=16)
    ap.add_argument("--eval-every", type=int, default=200)
    ap.add_argument("--max-steps", type=int, default=None)
    ap.add_argument("--resume", default=None, help="从检查点续训（实验04编辑训练）")
    ap.add_argument("--executor", default="small", choices=["small", "llm"],
                    help="Execute算子实现：small=4层小Transformer；llm=骨干自身计算")
    ap.add_argument("--node-dim", type=int, default=256,
                    help="节点状态维度；设为骨干隐层宽度即v5对齐表示空间")
    ap.add_argument("--node-tokens", type=int, default=2,
                    help="节点状态token数：2=池化(单位数)，5=逐位(多位数0-999)")
    ap.add_argument("--exec-layers", type=int, default=4,
                    help="执行器Transformer层数（加深攻多位数）")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, "config.json"), "w", encoding="utf-8") as f:
        json.dump(vars(args), f, ensure_ascii=False, indent=2)
    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    variants = tuple(args.variants.split(","))
    train_recs = C.load_records(os.path.join(args.data, "train.jsonl"),
                                variants, args.seed, args.limit)
    dev_recs = C.load_records(os.path.join(args.data, "dev.jsonl"),
                              variants, args.seed + 1,
                              min(200, args.limit or 200))
    print(f"train records: {len(train_recs)}, dev: {len(dev_recs)}", flush=True)

    tok = AutoTokenizer.from_pretrained(args.model)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    # fp32 主权重：辅助路径经骨干反传在 bf16 下产生 NaN（已验证），
    # 5090 32GB 足以容纳 1.7B fp32 训练；推理评估仍可用 bf16
    base = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.float32)
    cfg = LoraConfig(r=args.lora_r, lora_alpha=2 * args.lora_r, lora_dropout=0.05,
                     target_modules=["q_proj", "v_proj"], task_type="CAUSAL_LM")
    base = get_peft_model(base, cfg, adapter_name="encoder")
    base.add_adapter("reader", cfg)
    if args.executor == "llm":
        base.add_adapter("executor", cfg)
    base.to(device)
    model = InterLatModel(base, tok, node_dim=args.node_dim,
                          executor_type=args.executor, node_tokens=args.node_tokens,
                          exec_layers=args.exec_layers)
    model.to(device)
    # 新增模块统一 bf16 与骨干一致（优化器状态仍为 fp32）
    # 新增模块保持 fp32（与骨干一致）
    # 骨干冻结，仅 LoRA＋投影＋执行器＋状态头可训练
    for n, p in model.named_parameters():
        p.requires_grad = ("lora_" in n) or n.startswith(
            ("enc_proj", "read_proj", "executor", "state_head", "exec_proj", "exec_out"))
    # 冻结骨干嵌入表：辅助路径经全词表嵌入反传会在 bf16 下产生 NaN 梯度
    for n, p in model.named_parameters():
        if "embed_tokens" in n or n.endswith("lm_head.weight"):
            p.requires_grad = False
    if args.resume:
        rstep = load_trainable(model, args.resume, device)
        print(f"resumed from {args.resume} (step {rstep})", flush=True)
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.01)
    base.gradient_checkpointing_enable()
    base.enable_input_require_grads()

    def batches(recs, shuffle=True):
        import random
        rng = random.Random(args.seed)
        by_fam = C.group_by_family(recs)
        order = []
        for fam, rs in by_fam.items():
            if shuffle:
                rng.shuffle(rs)
            for i in range(0, len(rs), args.micro_batch):
                order.append((fam, rs[i:i + args.micro_batch]))
        if shuffle:
            rng.shuffle(order)
        return order

    @torch.no_grad()
    def evaluate(max_batches: int = 20) -> dict:
        model.eval()
        tot, correct, n = 0.0, 0, 0
        for fam, rs in batches(dev_recs, shuffle=False)[:max_batches]:
            g = C.collate(rs, device)
            z_in = model.encode_inputs(g.input_texts, device)
            z_all = model.execute_all(z_in, g)
            ids = model.read_answers(z_all, g, device, generate=True, max_new_tokens=8)
            texts = tok.batch_decode(ids, skip_special_tokens=True)
            for txt, r in zip(texts, rs):
                tot += 1
                n += 1
                # 首 token 精确匹配（答案为单 token 数值或 true/false）
                first = txt.strip().split()[0] if txt.strip() else ""
                if first == r["answer_text"].strip():
                    correct += 1
        model.train()
        return {"dev_answer_acc": correct / max(tot, 1), "dev_n": n}

    log = []
    step = 0
    t_start = time.time()
    # cosine 学习率衰减（稳定后期收敛，修复多节点链训练振荡）
    total_steps = args.epochs * max(1, len(train_recs) // args.micro_batch)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=total_steps, eta_min=args.lr * 0.1)
    model.train()
    for ep in range(args.epochs):
        for fam, rs in batches(train_recs):
            g = C.collate(rs, device)
            loss, parts = model(g, device, lambda_s=args.lambda_s,
                                lambda_aux=args.lambda_aux)
            (loss / args.grad_accum).backward()
            if (step + 1) % args.grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(params, 1.0)
                opt.step()
                opt.zero_grad()
                sched.step()
            step += 1
            if step % 20 == 0:
                elapsed = time.time() - t_start
                rec = {"epoch": ep, "step": step, "loss": float(loss.detach()),
                       "answer_loss": parts["answer"],
                       "seconds": round(elapsed, 1),
                       "samples_per_s": round(step * args.micro_batch / elapsed, 2)}
                if device == "cuda":
                    rec["vram_peak_gb"] = round(torch.cuda.max_memory_allocated() / 2**30, 2)
                log.append(rec)
                print(json.dumps(rec, ensure_ascii=False), flush=True)
            if step % args.eval_every == 0:
                ev = evaluate()
                ev.update({"step": step})
                log.append({"eval": ev})
                print("EVAL " + json.dumps(ev, ensure_ascii=False), flush=True)
                save_trainable(model, os.path.join(args.out, f"ckpt_step{step}.pt"), step)
            if args.max_steps and step >= args.max_steps:
                break
        if args.max_steps and step >= args.max_steps:
            break

    ev = evaluate(max_batches=50)
    save_trainable(model, os.path.join(args.out, "ckpt_final.pt"), step)
    summary = {"experiment": args.out, "final_eval": ev, "total_steps": step,
               "total_seconds": round(time.time() - t_start, 1),
               "train_records": len(train_recs), "variants": variants}
    with open(os.path.join(args.out, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    with open(os.path.join(args.out, "train_log.jsonl"), "w", encoding="utf-8") as f:
        for rec in log:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
