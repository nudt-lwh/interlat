"""CODI 对照（同骨干同数据复现，EMNLP 2025 配方）。

教师任务：任务文本＋显式计算轨迹→答案（交叉熵）；
学生任务：任务文本＋<bot>＋6 个连续思考（末层隐状态经 MLP+LN 投影喂回）＋<eot>→答案；
蒸馏：答案前冒号位置全层隐激活 L1 对齐（教师 stop-gradient，按教师批内 std 归一）。
与 InterLat 共享同一任务/真值/划分，保证公平。

用法：
  python -m interlat.training.train_codi --data experiment_data/runs/exp03_formal_data \
      --model experiment_data/artifacts/Qwen3-1.7B --out experiment_data/runs/exp03_codi_5090 \
      --epochs 2 --micro-batch 8 --grad-accum 8
"""
from __future__ import annotations

import argparse
import json
import os
import time

import torch
import torch.nn as nn
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer

from .train_cot import cot_text


class CODIModel(nn.Module):
    def __init__(self, backbone, tok, num_latent=6):
        super().__init__()
        self.backbone = backbone
        self.tok = tok
        self.num_latent = num_latent
        h = backbone.config.hidden_size
        self.prj = nn.Sequential(nn.Linear(h, h), nn.GELU(), nn.Linear(h, h),
                                 nn.LayerNorm(h))
        # 注册为 buffer：随 .to(device) 移动，且作为固定特殊嵌入不参与训练
        self.register_buffer("bot", torch.randn(1, 1, h) * 0.02)
        self.register_buffer("eot", torch.randn(1, 1, h) * 0.02)

    def _embed(self, ids):
        return self.backbone.get_input_embeddings()(ids)

    def forward(self, prompt_ids, prompt_mask, cot_ids, cot_mask,
                ans_ids, ans_mask, device):
        B = prompt_ids.shape[0]
        pe = self._embed(prompt_ids)
        # ---- 教师：prompt + CoT + 答案 ----
        ce = self._embed(cot_ids)
        ae = self._embed(ans_ids)
        t_seq = torch.cat([pe, ce, ae], dim=1)
        t_mask = torch.cat([prompt_mask, cot_mask, ans_mask], dim=1)
        t_labels = torch.cat([torch.full_like(prompt_ids, -100),
                              torch.where(cot_mask.bool(), cot_ids, -100),
                              torch.where(ans_mask.bool(), ans_ids, -100)], dim=1)
        t_out = self.backbone(inputs_embeds=t_seq, attention_mask=t_mask,
                              labels=t_labels, output_hidden_states=True)
        # 蒸馏位置：答案前最后一个 CoT token（近似冒号位置）
        t_pos = prompt_mask.sum(1) + cot_mask.sum(1) - 1
        t_h = torch.stack([t_out.hidden_states[l][torch.arange(B), t_pos]
                           for l in range(1, len(t_out.hidden_states))], 0)
        t_h = t_h.detach()
        # ---- 学生：prompt + bot + 连续思考 + eot + 答案 ----
        s_inputs = pe
        s_mask = prompt_mask
        cur = self.bot.expand(B, -1, -1)
        s_inputs = torch.cat([s_inputs, cur], dim=1)
        s_mask = torch.cat([s_mask, torch.ones(B, 1, device=device, dtype=torch.long)], 1)
        for _ in range(self.num_latent):
            out = self.backbone(inputs_embeds=s_inputs, attention_mask=s_mask,
                                output_hidden_states=True)
            h_last = out.hidden_states[-1][:, -1:]
            nxt = self.prj(h_last)
            s_inputs = torch.cat([s_inputs, nxt], dim=1)
            s_mask = torch.cat([s_mask, torch.ones(B, 1, device=device, dtype=torch.long)], 1)
        s_inputs = torch.cat([s_inputs, self.eot.expand(B, -1, -1)], dim=1)
        s_mask = torch.cat([s_mask, torch.ones(B, 1, device=device, dtype=torch.long)], 1)
        s_pos = s_inputs.shape[1] - 1
        s_full = torch.cat([s_inputs, ae], dim=1)
        s_full_mask = torch.cat([s_mask, ans_mask], dim=1)
        s_labels = torch.cat([torch.full((B, s_inputs.shape[1]), -100, device=device),
                              torch.where(ans_mask.bool(), ans_ids, -100)], dim=1)
        s_out = self.backbone(inputs_embeds=s_full, attention_mask=s_full_mask,
                              labels=s_labels, output_hidden_states=True)
        s_h = torch.stack([s_out.hidden_states[l][torch.arange(B), s_pos]
                           for l in range(1, len(s_out.hidden_states))], 0)
        std = t_h.std(dim=(1, 2), keepdim=True).clamp(min=1e-6)
        l_kd = ((s_h - t_h) / std).abs().mean()
        return t_out.loss, s_out.loss, l_kd

    @torch.no_grad()
    def generate_answer(self, prompt_ids, prompt_mask, device, max_new=8):
        B = prompt_ids.shape[0]
        s_inputs = self._embed(prompt_ids)
        s_mask = prompt_mask
        cur = self.bot.expand(B, -1, -1)
        s_inputs = torch.cat([s_inputs, cur], dim=1)
        s_mask = torch.cat([s_mask, torch.ones(B, 1, device=device, dtype=torch.long)], 1)
        for _ in range(self.num_latent):
            out = self.backbone(inputs_embeds=s_inputs, attention_mask=s_mask,
                                output_hidden_states=True)
            nxt = self.prj(out.hidden_states[-1][:, -1:])
            s_inputs = torch.cat([s_inputs, nxt], dim=1)
            s_mask = torch.cat([s_mask, torch.ones(B, 1, device=device, dtype=torch.long)], 1)
        s_inputs = torch.cat([s_inputs, self.eot.expand(B, -1, -1)], dim=1)
        s_mask = torch.cat([s_mask, torch.ones(B, 1, device=device, dtype=torch.long)], 1)
        return self.backbone.generate(inputs_embeds=s_inputs, attention_mask=s_mask,
                                      max_new_tokens=max_new, do_sample=False,
                                      pad_token_id=self.tok.pad_token_id)


def save_trainable(model, path: str, step: int) -> None:
    sd = {k: v for k, v in model.state_dict().items()
          if ("lora_" in k) or k.startswith(("prj", "bot", "eot"))}
    torch.save({"model": sd, "step": step}, path)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--micro-batch", type=int, default=8)
    ap.add_argument("--grad-accum", type=int, default=8)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--lora-r", type=int, default=16)
    ap.add_argument("--beta", type=float, default=1.0)
    ap.add_argument("--gamma", type=float, default=1.0)
    ap.add_argument("--eval-every", type=int, default=500)
    ap.add_argument("--max-steps", type=int, default=None)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, "config.json"), "w", encoding="utf-8") as f:
        json.dump(vars(args), f, ensure_ascii=False, indent=2)
    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    def load(split):
        recs = []
        with open(os.path.join(args.data, f"{split}.jsonl"), encoding="utf-8") as f:
            for line in f:
                s = json.loads(line)
                p, t = cot_text(s, "original")
                cot, ans = t.rsplit("\n答案: ", 1)
                recs.append((p, cot + "\n", ans.strip()))
                if args.limit and len(recs) >= args.limit:
                    break
        return recs[:args.limit] if args.limit else recs

    train_recs, dev_recs = load("train"), load("dev")[:200]
    print(f"train {len(train_recs)}, dev {len(dev_recs)}", flush=True)

    tok = AutoTokenizer.from_pretrained(args.model, padding_side="left")
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    base = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16)
    cfg = LoraConfig(r=args.lora_r, lora_alpha=2 * args.lora_r, lora_dropout=0.05,
                     target_modules=["q_proj", "v_proj"], task_type="CAUSAL_LM")
    base = get_peft_model(base, cfg)
    base.to(device)
    base.gradient_checkpointing_enable()
    base.enable_input_require_grads()
    model = CODIModel(base, tok)
    model.to(device)
    model.prj.to(torch.bfloat16)
    model.bot = model.bot.to(torch.bfloat16)
    model.eot = model.eot.to(torch.bfloat16)
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.01)

    def make_batch(recs):
        ps, cs, as_ = zip(*recs)
        pe = tok(list(ps), return_tensors="pt", padding=True, truncation=True, max_length=160).to(device)
        ce = tok(list(cs), return_tensors="pt", padding=True, truncation=True, max_length=192).to(device)
        ae = tok([a + tok.eos_token for a in as_], return_tensors="pt", padding=True,
                 truncation=True, max_length=16).to(device)
        return pe, ce, ae

    @torch.no_grad()
    def evaluate(max_batches=10):
        model.eval()
        correct = tot = 0
        for i in range(0, min(len(dev_recs), max_batches * args.micro_batch), args.micro_batch):
            recs = dev_recs[i:i + args.micro_batch]
            pe, _, _ = make_batch(recs)
            gen = model.generate_answer(pe["input_ids"], pe["attention_mask"], device)
            texts = tok.batch_decode(gen, skip_special_tokens=True)
            for txt, (_, _, a) in zip(texts, recs):
                tot += 1
                first = txt.strip().split()[0] if txt.strip() else ""
                if first == a:
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
            pe, ce, ae = make_batch(recs)
            lt, ls, lkd = model(pe["input_ids"], pe["attention_mask"],
                                ce["input_ids"], ce["attention_mask"],
                                ae["input_ids"], ae["attention_mask"], device)
            loss = ls + args.beta * lkd + args.gamma * lt
            (loss / args.grad_accum).backward()
            if (step + 1) % args.grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(params, 1.0)
                opt.step()
                opt.zero_grad()
            step += 1
            if step % 20 == 0:
                el = time.time() - t_start
                rec = {"epoch": ep, "step": step, "loss": float(loss.detach()),
                       "student": float(ls.detach()), "kd": float(lkd.detach()),
                       "teacher": float(lt.detach()), "seconds": round(el, 1),
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
