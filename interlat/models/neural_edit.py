"""神经工作空间的编辑与依赖驱动局部重执行（C1/C3 的神经实现）。

与符号版 workspace/edit.py 接口对齐：
- 编辑输入节点：新值经同一编码器重新 Encode，替换目标节点状态；
- 受影响后代按拓扑序用 Execute 重算（首版重算全部后代）；
- 未受影响节点保持原 tensor 与原版本；
- 读取器每次基于当前工作空间新建前缀（不复用旧 KV）。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch

from .workspace_model import InterLatModel, NODE_TOKENS


@dataclass
class NeuralWorkspace:
    z: torch.Tensor                    # (1, N, T, D) 连续状态
    versions: list[int]
    topo: list[int]                    # 拓扑序节点索引
    parents: list[tuple[int, ...]]
    roles: list[tuple[int, ...]]
    ops: list[int]
    input_mask: list[bool]
    family: str
    entity: str
    names: list[str]

    @staticmethod
    def build(model: InterLatModel, g, device, layout: dict) -> "NeuralWorkspace":
        """g: 单样本 BatchGraph；layout: collate.family_layout 输出。"""
        z_in = model.encode_inputs(g.input_texts, device)      # (1, n_in, T, D)
        z_all = model.execute_all(z_in, g)                     # (1, N, T, D)
        N = z_all.shape[1]
        n_in = len(layout["names"]) - len(layout["derived"])
        parents: list[tuple[int, ...]] = [()] * n_in
        roles: list[tuple[int, ...]] = [()] * n_in
        ops: list[int] = [0] * n_in
        for j, (op, ps, rs) in enumerate(layout["derived"]):
            parents.append(tuple(ps))
            roles.append(tuple(rs))
            ops.append(op)
        return NeuralWorkspace(
            z=z_all, versions=[0] * N,
            topo=list(range(N)),           # 生成器保证输入在前、派生按依赖序
            parents=parents, roles=roles, ops=ops,
            input_mask=[i < n_in for i in range(N)],
            family=layout.get("family", ""), entity="",
            names=layout["names"],
        )

    def descendants(self, root: int) -> set[int]:
        children: dict[int, list[int]] = {i: [] for i in range(len(self.topo))}
        for i, ps in enumerate(self.parents):
            for p in ps:
                children[p].append(i)
        seen: set[int] = set()
        stack = [root]
        while stack:
            cur = stack.pop()
            for ch in children[cur]:
                if ch not in seen:
                    seen.add(ch)
                    stack.append(ch)
        return seen


def neural_edit(model: InterLatModel, ws: NeuralWorkspace, target_name: str,
                new_value: Any, device, op: str = "set") -> dict:
    """编辑输入节点并局部重执行，返回统计（重算数/保持数/受影响集合）。"""
    idx = ws.names.index(target_name)
    if not ws.input_mask[idx]:
        raise ValueError("first version only edits input nodes")
    old_z = ws.z.detach().clone()
    old_versions = list(ws.versions)

    if op == "add":
        raise NotImplementedError("add 编辑在神经版转换为新增更新节点，第二阶段实现")
    text = f"{target_name} = {new_value}"
    z_new = model.encode_inputs([[text]], device)[0, 0]        # (T, D)
    ws.z[0, idx] = z_new
    ws.versions[idx] += 1

    affected = ws.descendants(idx)
    ex = model.executor
    recomputed = 0
    for j in ws.topo:
        if j not in affected:
            continue
        ps = ws.parents[j]
        pz = ws.z[0, list(ps)].unsqueeze(0)                    # (1, P, T, D)
        roles = torch.tensor([ws.roles[j]], device=device)
        x = ex.embed_parents(pz, roles)
        ops_t = torch.tensor([ws.ops[j]], device=device)
        ws.z[0, j] = ex.forward_single(x, ops_t)[0]
        ws.versions[j] += 1
        recomputed += 1

    preserved = 0
    for i in range(len(ws.topo)):
        if i != idx and i not in affected:
            if ws.versions[i] == old_versions[i] and torch.equal(ws.z[0, i], old_z[0, i]):
                preserved += 1
    return {"target": target_name, "recomputed_nodes": recomputed,
            "preserved_nodes": preserved, "affected": sorted(affected)}


class ReaderKVCache:
    """P0-3：查询 token 的 KV 复用＋输出节点状态前缀按版本精确失效。

    前缀结构＝[查询文本 KV][状态前缀 KV]。查询不变时复用查询 KV；
    状态前缀以 (query, out_names, 输出版本元组) 为 key，版本变化才重算。
    不复用旧输出节点的 KV。
    """

    def __init__(self):
        self.q = {}   # cond_text -> (past_key_values, q_len)
        self.s = {}   # (cond_text, out_key, ver_tuple) -> (past_key_values, total_len)

    def read(self, model, ws, query_text, out_names, device, max_new_tokens=16):
        from ..models.workspace_model import BatchGraph  # noqa: F401
        model.backbone.set_adapter("reader")
        out_idx = [ws.names.index(n) for n in out_names]
        ver = tuple(ws.versions[i] for i in out_idx)
        cond = f"{query_text} 输出节点: {', '.join(out_names)}"
        if cond not in self.q:
            q = model.tok([cond], return_tensors="pt", padding=True,
                          truncation=True, max_length=64).to(device)
            qemb = model.backbone.get_input_embeddings()(q["input_ids"])
            outq = model.backbone(inputs_embeds=qemb,
                                  attention_mask=q["attention_mask"], use_cache=True)
            self.q[cond] = (outq.past_key_values, qemb.shape[1])
        qpast, qlen = self.q[cond]
        key = (cond, tuple(out_names), ver)
        if key not in self.s:
            out_z = ws.z[0, out_idx].unsqueeze(0)             # (1,n_out,T,D)
            state_prefix = model.read_proj(out_z.flatten(2))  # (1,n_out*T,hidden)
            L = qlen + state_prefix.shape[1]
            mask = torch.ones(1, L, device=device, dtype=torch.long)
            pos = torch.arange(qlen, L, device=device).unsqueeze(0)
            outs = model.backbone(inputs_embeds=state_prefix, attention_mask=mask,
                                  past_key_values=qpast, position_ids=pos, use_cache=True)
            self.s[key] = (outs.past_key_values, L)
        spast, L = self.s[key]
        # generate 会原地扩展 past_key_values；克隆后再喂，避免污染缓存条目
        from transformers.cache_utils import DynamicCache
        clone = DynamicCache()
        for li, layer in enumerate(spast.layers):
            clone.update(layer.keys.clone(), layer.values.clone(), li)
        gen = model.backbone.generate(past_key_values=clone,
                                      attention_mask=torch.ones(1, L, device=device, dtype=torch.long),
                                      max_new_tokens=max_new_tokens, do_sample=False,
                                      pad_token_id=model.tok.pad_token_id)
        return model.tok.batch_decode(gen, skip_special_tokens=True)[0]


def read_query(model: InterLatModel, ws: NeuralWorkspace, query_text: str,
               out_names: list[str], device, max_new_tokens: int = 16) -> str:
    """基于当前工作空间新建连续前缀读取答案（不复用旧 KV）。"""
    from ..models.workspace_model import BatchGraph
    out_idx = torch.tensor([[ws.names.index(n) for n in out_names]], device=device)
    g = BatchGraph(input_texts=[[]], derived_ops=torch.zeros(1, 0, dtype=torch.long, device=device),
                   derived_parents=torch.zeros(1, 0, 1, dtype=torch.long, device=device),
                   derived_roles=torch.zeros(1, 0, 1, dtype=torch.long, device=device),
                   node_types=[[]], query_texts=[query_text],
                   answer_texts=None, query_out_idx=out_idx,
                   query_out_names=[out_names])
    ids = model.read_answers(ws.z, g, device, max_new_tokens=max_new_tokens, generate=True)
    return model.tok.batch_decode(ids, skip_special_tokens=True)[0]
