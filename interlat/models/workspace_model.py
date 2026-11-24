"""Encode local facts, execute graph operators and read answers from node states."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn

OP_LIST = ("input", "add", "sub", "mul", "floordiv", "gt", "lt", "min", "max", "select")
OP_INDEX = {o: i for i, o in enumerate(OP_LIST)}
ROLE_LIST = ("lhs", "rhs", "num", "den", "condition", "then", "else")
ROLE_INDEX = {r: i for i, r in enumerate(ROLE_LIST)}

NODE_DIM = 256
MAX_DIGITS = 4          # 最多 4 位数字
NODE_TOKENS = 2         # v9 架构（EOS+池化）：加法97%/减法94.5%可靠；v10逐位(5)已证明退步
DIGIT_EOS = 10          # 数字序列终止符（消除右对齐 PAD 歧义）
DIGIT_PAD = 11          # EOS 之后的填充（损失忽略）
DIGIT_VOCAB = 12        # 0-9 + EOS(10) + PAD(11)
SIGN_CLASSES = 2        # 0:非负 1:负


import os as _os
# 消融开关（P0-6）：IL_LABEL_ALIGN=right_pad 时用旧 v8 右对齐 PAD 口径（无 EOS）。
# 默认 left_eos（v9），保证已有 run 可复现。仅消融 run 设置该环境变量。
LABEL_ALIGN = _os.environ.get("IL_LABEL_ALIGN", "left_eos")


def value_to_labels(v) -> tuple[int, list[int]]:
    """Encode a value as sign and digit labels with the selected EOS and padding convention."""
    if LABEL_ALIGN == "right_pad":
        if isinstance(v, bool):
            return 0, [DIGIT_PAD] * (MAX_DIGITS - 1) + [int(v)]
        sign = 1 if v < 0 else 0
        digits = [int(c) for c in str(abs(v))][:MAX_DIGITS]
        seq = [DIGIT_PAD] * (MAX_DIGITS - len(digits)) + digits
        return sign, seq
    if isinstance(v, bool):
        return 0, [int(v), DIGIT_EOS] + [DIGIT_PAD] * (MAX_DIGITS - 2)
    sign = 1 if v < 0 else 0
    digits = [int(c) for c in str(abs(v))][:MAX_DIGITS - 1]
    seq = digits + [DIGIT_EOS]
    seq = seq + [DIGIT_PAD] * (MAX_DIGITS - len(seq))
    return sign, seq


def decode_digits(seq: list[int]) -> str:
    """Decode digit labels, stopping at EOS or skipping leading padding as appropriate."""
    if LABEL_ALIGN == "right_pad":
        return "".join(str(d) for d in seq if d != DIGIT_PAD)
    out = []
    for d in seq:
        if d == DIGIT_EOS or d == DIGIT_PAD:
            break
        out.append(str(d))
    return "".join(out)


class StateHead(nn.Module):
    """Read signs and digits from pooled or position-specific node states."""

    def __init__(self, dim: int = NODE_DIM, node_tokens: int = NODE_TOKENS):
        super().__init__()
        self.node_tokens = node_tokens
        self.per_digit = node_tokens > 2
        if self.per_digit:
            self.sign = nn.Sequential(nn.Linear(dim, dim), nn.GELU(), nn.Linear(dim, SIGN_CLASSES))
            self.digit = nn.Sequential(nn.Linear(dim, dim), nn.GELU(), nn.Linear(dim, DIGIT_VOCAB))
            self.bool_head = nn.Sequential(nn.Linear(dim, dim), nn.GELU(), nn.Linear(dim, 2))
        else:
            self.pool = nn.Sequential(
                nn.Linear(dim * node_tokens, dim), nn.GELU(), nn.Linear(dim, dim))
            self.sign = nn.Linear(dim, SIGN_CLASSES)
            self.digits = nn.Linear(dim, MAX_DIGITS * DIGIT_VOCAB)
            self.bool_head = nn.Linear(dim, 2)

    def forward(self, z: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # z: (B, node_tokens, dim)
        if self.per_digit:
            sign_logits = self.sign(z[:, 0])
            digit_logits = self.digit(z[:, 1:1 + MAX_DIGITS])
            bool_logits = self.bool_head(z[:, 0])
            return sign_logits, digit_logits, bool_logits
        h = self.pool(z.flatten(1))
        return self.sign(h), self.digits(h).view(-1, MAX_DIGITS, DIGIT_VOCAB), self.bool_head(h)


class Executor(nn.Module):
    """Apply a shared update conditioned on the operator and parent roles."""

    def __init__(self, dim: int = NODE_DIM, layers: int = 4, heads: int = 4,
                 node_tokens: int = NODE_TOKENS):
        super().__init__()
        self.dim = dim
        self.node_tokens = node_tokens
        self.op_emb = nn.Embedding(len(OP_LIST), dim)
        self.role_emb = nn.Embedding(len(ROLE_LIST), dim)
        layer = nn.TransformerEncoderLayer(dim, heads, dim * 2, batch_first=True,
                                           norm_first=True)
        self.net = nn.TransformerEncoder(layer, layers)
        self.query = nn.Parameter(torch.randn(node_tokens, dim) * 0.02)

    def embed_parents(self, parent_z: torch.Tensor, roles: torch.Tensor) -> torch.Tensor:
        """Add role embeddings and flatten parents and state tokens into a single sequence."""
        B, P = parent_z.shape[:2]
        return parent_z.flatten(1, 2) + self.role_emb(roles).unsqueeze(2).repeat(
            1, 1, self.node_tokens, 1).flatten(1, 2)

    def forward_single(self, x_parents: torch.Tensor, ops: torch.Tensor) -> torch.Tensor:
        """Execute one node from role-embedded parent states."""
        B = x_parents.shape[0]
        q = self.query.unsqueeze(0).expand(B, -1, -1) + self.op_emb(ops).unsqueeze(1)
        out = self.net(torch.cat([q, x_parents], dim=1))
        return out[:, :self.node_tokens]

    def forward(self, parent_z: torch.Tensor, ops: torch.Tensor,
                roles: torch.Tensor) -> torch.Tensor:
        # parent_z: (B, P, node_tokens, dim); ops: (B,); roles: (B, P)
        B = parent_z.shape[0]
        x = self.embed_parents(parent_z, roles)
        q = self.query.unsqueeze(0).expand(B, -1, -1) + self.op_emb(ops).unsqueeze(1)
        x = torch.cat([q, x], dim=1)
        out = self.net(x)
        return out[:, :self.node_tokens]


@dataclass
class BatchGraph:
    """Store a batch of topologically ordered graphs with input nodes first."""
    input_texts: list[list[str]]        # B×n_in 局部事实
    derived_ops: torch.Tensor           # (B, n_der)
    derived_parents: torch.Tensor       # (B, n_der, P) 全局节点索引
    derived_roles: torch.Tensor         # (B, n_der, P)
    node_types: list[list[str]]         # B×N
    state_labels: Optional[torch.Tensor] = None   # (B, N, 1+MAX_DIGITS)
    query_texts: Optional[list[str]] = None
    answer_texts: Optional[list[str]] = None
    query_out_idx: Optional[torch.Tensor] = None  # (B, n_out)
    query_out_names: Optional[list[list[str]]] = None  # 输出节点结构标签


class InterLatModel(nn.Module):
    def __init__(self, backbone, tokenizer, node_dim: int = NODE_DIM,
                 executor_type: str = "small", node_tokens: int = NODE_TOKENS,
                 exec_layers: int = 4, exec_dim: int = 0):
        super().__init__()
        self.backbone = backbone        # PeftModel，含 encoder/reader/executor adapter
        self.tok = tokenizer
        hidden = backbone.config.hidden_size
        self.node_tokens = node_tokens   # 实例级：单位数2（池化），多位数5（逐位）
        self.enc_proj = nn.Linear(hidden, node_dim)
        self.read_proj = nn.Linear(node_dim * node_tokens, hidden)
        # 执行器层数可配置（加深攻多位数）；宽度保持 node_dim 以与状态一致
        self.executor = Executor(node_dim, layers=exec_layers, node_tokens=node_tokens)
        self.state_head = StateHead(node_dim, node_tokens=node_tokens)
        self.node_dim = node_dim
        self.hidden = hidden
        self.executor_type = executor_type
        if executor_type == "llm":
            self.exec_proj = nn.Linear(node_dim * node_tokens, hidden)
            self.exec_out = nn.Linear(hidden, node_dim * node_tokens)

    def execute_llm(self, parent_z: torch.Tensor, op_ids: torch.Tensor,
                    device) -> torch.Tensor:
        """Execute nodes with the backbone using parent states and operator IDs."""
        B, P = parent_z.shape[:2]
        self.backbone.set_adapter("executor")
        prefix = self.exec_proj(parent_z.flatten(2))          # (B, P*T, hidden)
        op_texts = [OP_LIST[int(o)] for o in op_ids]
        ot = self.tok(op_texts, return_tensors="pt", padding=True).to(device)
        oemb = self.backbone.get_input_embeddings()(ot["input_ids"])
        full = torch.cat([prefix, oemb], dim=1)
        mask = torch.cat([torch.ones(B, prefix.shape[1], device=device, dtype=torch.long),
                          ot["attention_mask"]], dim=1)
        out = self.backbone(inputs_embeds=full, attention_mask=mask,
                            output_hidden_states=True)
        h = out.hidden_states[-1]
        m = mask.unsqueeze(-1)
        pooled = (h * m).sum(1) / m.sum(1).clamp(min=1)
        return self.exec_out(pooled).view(B, self.node_tokens, self.node_dim)

    def encode_inputs(self, texts: list[list[str]], device) -> torch.Tensor:
        """Encode local facts into pooled or digit-aligned input states."""
        B = len(texts)
        flat = [t for row in texts for t in row]
        enc = self.tok(flat, return_tensors="pt", padding=True,
                       truncation=True, max_length=32).to(device)
        self.backbone.set_adapter("encoder")
        emb = self.backbone.get_input_embeddings()(enc["input_ids"])
        out = self.backbone(inputs_embeds=emb,
                            attention_mask=enc["attention_mask"],
                            output_hidden_states=True)
        h = out.hidden_states[-1]                       # (B*n_in, L, hidden)
        mask = enc["attention_mask"]
        if self.node_tokens <= 2:
            pooled = (h * mask.unsqueeze(-1)).sum(1) / mask.sum(1).clamp(min=1).unsqueeze(-1)
            zp = self.enc_proj(pooled)                       # (B*n_in, dim)
            z = zp.unsqueeze(1).repeat(1, self.node_tokens, 1)  # (B*n_in, node_tokens, dim)
            return z.view(B, len(texts[0]), self.node_tokens, self.node_dim)
        # 逐位：取每个样本末尾 MAX_DIGITS 个有效 token 的隐藏状态（数字位）
        n_in = len(texts[0])
        seq_len = h.shape[1]
        lengths = mask.sum(1)                            # 每样本有效长度
        # 数字位索引：从末尾往前 MAX_DIGITS 个（不足补0向量）
        digit_h = []
        for i in range(h.shape[0]):
            L = int(lengths[i].item())
            idxs = list(range(max(0, L - MAX_DIGITS), L))   # 末尾数字token
            hs = h[i, idxs]                              # (<=MAX_DIGITS, hidden)
            if hs.shape[0] < MAX_DIGITS:
                pad = torch.zeros(MAX_DIGITS - hs.shape[0], hs.shape[1],
                                  device=device, dtype=hs.dtype)
                hs = torch.cat([pad, hs], 0)             # 左补0对齐
            digit_h.append(hs)
        digit_h = torch.stack(digit_h, 0)                # (B*n_in, MAX_DIGITS, hidden)
        # 符号位用整句均值（显式扩展 mask 避免广播歧义）
        m = mask.unsqueeze(-1).to(h.dtype).expand(h.shape)
        pooled = (h * m).sum(1) / mask.sum(1).clamp(min=1).unsqueeze(-1).to(h.dtype)
        sign_h = pooled.unsqueeze(1)                     # (B*n_in, 1, hidden)
        full_h = torch.cat([sign_h, digit_h], 1)         # (B*n_in, 1+MAX_DIGITS, hidden)
        z = self.enc_proj(full_h)                        # (B*n_in, node_tokens, dim)
        return z.view(B, n_in, self.node_tokens, self.node_dim)

    def execute_all(self, z_in: torch.Tensor, g: BatchGraph) -> torch.Tensor:
        """Evaluate derived nodes in topological order and return the full state tensor."""
        B, n_in = z_in.shape[:2]
        states = list(z_in.unbind(dim=1))
        n_der = g.derived_ops.shape[1]
        for j in range(n_der):
            cur = torch.stack(states, dim=1)                  # (B, curN, T, D)
            pidx = g.derived_parents[:, j]                    # (B, P)
            idx = pidx[:, :, None, None].expand(B, pidx.shape[1],
                                                self.node_tokens, self.node_dim)
            pz = cur.gather(1, idx)                           # (B, P, T, D)
            if self.executor_type == "llm":
                zj = self.execute_llm(pz, g.derived_ops[:, j], z_in.device)
            else:
                zj = self.executor(pz, g.derived_ops[:, j], g.derived_roles[:, j])
            states.append(zj)
        return torch.stack(states, dim=1)

    def state_loss(self, z_all: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        B, N = labels.shape[:2]
        sign_logits, digit_logits, bool_logits = self.state_head(
            z_all.reshape(B * N, self.node_tokens, self.node_dim))
        sign_logits = sign_logits.float()
        digit_logits = digit_logits.float()
        bool_logits = bool_logits.float()
        sign_lab = labels[..., 0].reshape(B * N)
        dig_lab = labels[..., 1:].reshape(B * N, MAX_DIGITS)
        ce = nn.functional.cross_entropy
        loss = ce(sign_logits, sign_lab)
        # v9：数字词表 DIGIT_VOCAB=12，PAD(11) 忽略，EOS(10) 参与监督
        loss = loss + ce(digit_logits.reshape(-1, DIGIT_VOCAB), dig_lab.reshape(-1),
                         ignore_index=DIGIT_PAD)
        # 布尔节点（首位 0/1 且次位 EOS）监督 true/false 读出
        is_bool = (dig_lab[:, 1] == DIGIT_EOS) & (dig_lab[:, 0] <= 1)
        if is_bool.any():
            loss = loss + ce(bool_logits[is_bool], dig_lab[is_bool, 0])
        return loss

    def read_answers(self, z_all: torch.Tensor, g: BatchGraph, device,
                     max_new_tokens: int = 16, generate: bool = False):
        """Compute loss or generate answer tokens from the query and output states."""
        self.backbone.set_adapter("reader")
        B = z_all.shape[0]
        out_z = torch.stack([z_all[b, g.query_out_idx[b]] for b in range(B)], 0)
        state_prefix = self.read_proj(out_z.flatten(2))       # (B, n_out*T, hidden)
        # Append the queried output node names.
        if g.query_out_names is not None:
            cond_texts = [f"{q} Output nodes: {', '.join(names)}"
                          for q, names in zip(g.query_texts, g.query_out_names)]
        else:
            cond_texts = list(g.query_texts)
        q = self.tok(cond_texts, return_tensors="pt", padding=True,
                     truncation=True, max_length=64).to(device)
        qemb = self.backbone.get_input_embeddings()(q["input_ids"])
        full = torch.cat([qemb, state_prefix], dim=1)
        pmask = torch.ones(B, state_prefix.shape[1], device=device, dtype=torch.long)
        mask = torch.cat([q["attention_mask"], pmask], dim=1)
        if generate:
            return self.backbone.generate(inputs_embeds=full, attention_mask=mask,
                                          max_new_tokens=max_new_tokens, do_sample=False,
                                          pad_token_id=self.tok.pad_token_id)
        a = self.tok(g.answer_texts, return_tensors="pt", padding=True,
                     truncation=True, max_length=24).to(device)
        # Terminate each answer with EOS.
        eos = torch.full((B, 1), self.tok.eos_token_id, device=device, dtype=torch.long)
        aids = torch.cat([a["input_ids"], eos], dim=1)
        amask = torch.cat([a["attention_mask"], torch.ones(B, 1, device=device, dtype=torch.long)], dim=1)
        aemb = self.backbone.get_input_embeddings()(aids)
        seq = torch.cat([full, aemb], dim=1)
        smask = torch.cat([mask, amask], dim=1)
        labels = aids.clone()
        labels[amask == 0] = -100
        full_labels = torch.cat([torch.full((B, full.shape[1]), -100, device=device),
                                 labels], dim=1)
        out = self.backbone(inputs_embeds=seq, attention_mask=smask, labels=full_labels)
        return out.loss

    def read_answers_from_soft(self, soft_prefix: torch.Tensor, g: BatchGraph,
                               device) -> torch.Tensor:
        """Train the answer reader from differentiable state-head predictions."""
        self.backbone.set_adapter("reader")
        B = soft_prefix.shape[0]
        prefix = soft_prefix
        q = self.tok(g.query_texts, return_tensors="pt", padding=True,
                     truncation=True, max_length=48).to(device)
        qemb = self.backbone.get_input_embeddings()(q["input_ids"])
        full = torch.cat([prefix, qemb], dim=1)
        pmask = torch.ones(B, prefix.shape[1], device=device, dtype=torch.long)
        mask = torch.cat([pmask, q["attention_mask"]], dim=1)
        a = self.tok(g.answer_texts, return_tensors="pt", padding=True,
                     truncation=True, max_length=24).to(device)
        eos = torch.full((B, 1), self.tok.eos_token_id, device=device, dtype=torch.long)
        aids = torch.cat([a["input_ids"], eos], dim=1)
        amask = torch.cat([a["attention_mask"],
                           torch.ones(B, 1, device=device, dtype=torch.long)], dim=1)
        aemb = self.backbone.get_input_embeddings()(aids)
        seq = torch.cat([full, aemb], dim=1)
        smask = torch.cat([mask, amask], dim=1)
        labels = aids.clone()
        labels[amask == 0] = -100
        full_labels = torch.cat([torch.full((B, full.shape[1]), -100, device=device),
                                 labels], dim=1)
        out = self.backbone(inputs_embeds=seq, attention_mask=smask, labels=full_labels)
        return out.loss

    def forward(self, g: BatchGraph, device, lambda_s: float = 0.5,
                lambda_aux: float = 0.0):
        z_in = self.encode_inputs(g.input_texts, device)
        z_all = self.execute_all(z_in, g)
        loss_ans = self.read_answers(z_all, g, device)
        loss = loss_ans
        parts = {"answer": float(loss_ans.detach())}
        if g.state_labels is not None:
            ls = self.state_loss(z_all, g.state_labels.to(device))
            loss = loss + lambda_s * ls
            parts["state"] = float(ls.detach())
        if lambda_aux > 0 and g.answer_texts is not None:
            # 可微软读出：输出节点状态→(符号/数字/布尔)概率→加权嵌入前缀
            B = z_all.shape[0]
            out_z = torch.stack([z_all[b, g.query_out_idx[b]] for b in range(B)], 0)
            n_out = out_z.shape[1]
            sign_logits, digit_logits, bool_logits = self.state_head(
                out_z.reshape(-1, self.node_tokens, self.node_dim))
            emb = self.backbone.get_input_embeddings().weight
            # 全部数字位软嵌入（0..9，PAD→空串占位用'0'）＋符号位
            dig_ids = torch.tensor(
                [self.tok.encode(str(d), add_special_tokens=False)[0] if d < 10
                 else self.tok.encode("0", add_special_tokens=False)[0]
                 for d in range(11)], device=device)
            p_dig = torch.softmax(digit_logits, dim=-1)            # (B*n_out, MAX_DIGITS, 11)
            dig_emb = emb[dig_ids]                                 # (11, hidden)
            soft_dig = torch.einsum("bkv,vh->bkh", p_dig, dig_emb)
            # (B*n_out, MAX_DIGITS, hidden)
            # 首版答案均非负，符号位用零占位（负数扩展时再引入符号嵌入）
            soft_sign = torch.zeros(B * n_out, 1, soft_dig.shape[2],
                                    device=device, dtype=soft_dig.dtype)
            soft_num = torch.cat([soft_sign, soft_dig], dim=1)     # (B*n_out,1+MAX_DIGITS,hidden)
            # 布尔软嵌入（true/false）
            tf_ids = torch.tensor(
                [self.tok.encode("true", add_special_tokens=False)[0],
                 self.tok.encode("false", add_special_tokens=False)[0]], device=device)
            p_bool = torch.softmax(bool_logits, dim=-1)
            soft_bool = (p_bool @ emb[tf_ids]).unsqueeze(1)        # (B*n_out,1,hidden)
            # 按答案文本选择布尔/数字前缀（训练时已知答案类型）
            is_bool = torch.tensor(
                [a.strip() in ("true", "false") for a in g.answer_texts], device=device)
            L = soft_num.shape[1]
            soft_bool_pad = torch.cat([soft_bool, torch.zeros(
                B * n_out, L - 1, soft_num.shape[2], device=device, dtype=soft_num.dtype)], dim=1)
            soft = torch.where(is_bool.repeat_interleave(n_out).view(B, n_out, 1, 1),
                               soft_bool_pad.view(B, n_out, L, -1),
                               soft_num.view(B, n_out, L, -1))
            soft = soft.flatten(1, 2)                              # (B, n_out*L, hidden)
            la = self.read_answers_from_soft(soft, g, device)
            loss = loss + lambda_aux * la
            parts["aux_answer"] = float(la.detach())
        return loss, parts
