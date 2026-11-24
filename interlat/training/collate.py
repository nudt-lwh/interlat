"""把生成器样本转为 InterLatModel 的 BatchGraph。

同一结构家族内节点骨架一致（输入顺序按 FAMILIES 定义固定，节点重命名由
实体前缀承担），因此可按家族组 batch，父节点索引跨样本相同。
"""
from __future__ import annotations

import random
from typing import Any

import torch

from ..data.generator import FAMILIES
from ..models.workspace_model import (BatchGraph, OP_INDEX, ROLE_INDEX,
                                       value_to_labels)
from ..workspace.schema import ARG_ROLES


def family_layout(family: str) -> dict[str, Any]:
    if family.startswith("simple_"):
        op = family[len("simple_"):]
        names = ["a", "b", "c"]
        der = [(OP_INDEX[op], [0, 1], [ROLE_INDEX["lhs"], ROLE_INDEX["rhs"]])]
        return {"names": names, "n_in": 2, "derived": der,
                "queries": {"result": ("c",)}}
    if family == "chain5":
        names = ["a", "b", "d", "f", "c", "e", "g"]
        idx = {n: i for i, n in enumerate(names)}
        der = [(OP_INDEX["add"], [idx["a"], idx["b"]], [ROLE_INDEX["lhs"], ROLE_INDEX["rhs"]]),
               (OP_INDEX["add"], [idx["c"], idx["d"]], [ROLE_INDEX["lhs"], ROLE_INDEX["rhs"]]),
               (OP_INDEX["add"], [idx["e"], idx["f"]], [ROLE_INDEX["lhs"], ROLE_INDEX["rhs"]])]
        return {"names": names, "n_in": 4, "derived": der,
                "queries": {"result": ("g",)}}
    spec = FAMILIES[family]
    names = list(spec["inputs"]) + [d[0] for d in spec["derived"]]
    idx = {n: i for i, n in enumerate(names)}
    der = []
    for name, op, parents in spec["derived"]:
        der.append((OP_INDEX[op], [idx[p] for p in parents],
                    [ROLE_INDEX[r] for r in ARG_ROLES[op]]))
    return {"names": names, "n_in": len(spec["inputs"]), "derived": der,
            "queries": spec["queries"]}


def sample_to_record(sample: dict, variant: str, rng: random.Random) -> dict:
    """variant: original / single0 / single1 / combo。返回训练记录。"""
    fam = sample["structure_family"]
    lay = family_layout(fam)
    ent = sample["entity"]
    ref = sample["reference"]
    if variant == "original":
        states = ref["original"]["states"]
        answers = ref["original"]["answers"]
    elif variant.startswith("single"):
        i = int(variant[-1])
        states = ref["single"][i]["states"]
        answers = ref["single"][i]["answers"]
    else:
        states = ref["combo_forward"]["states"]
        answers = ref["combo_forward"]["answers"]

    input_texts = []
    state_labels = []
    node_types = []
    if fam.startswith("simple_"):
        input_names = ["a", "b"]
    elif fam == "chain5":
        input_names = ["a", "b", "d", "f"]
    else:
        input_names = FAMILIES[fam]["inputs"]
    for name in lay["names"]:
        nid = f"{ent}.{name}"
        v = states[nid]
        if name in input_names:
            input_texts.append(f"{name} = {v}")
        sign, digits = value_to_labels(v)
        state_labels.append([sign] + digits)
        node_types.append("bool" if isinstance(v, bool) else "num")

    # 随机选一个查询训练（多查询复用由工作空间结构保证）
    qname = rng.choice(list(lay["queries"]))
    out_names = lay["queries"][qname]
    qtext = f"{qname}?"
    vals = answers[qname]
    atext = ", ".join("true" if v is True else "false" if v is False else str(v)
                      for v in vals)
    return {
        "family": fam,
        "input_texts": input_texts,
        "state_labels": state_labels,
        "query_text": qtext,
        "answer_text": atext,
        "query_out_idx": [lay["names"].index(n) for n in out_names],
        "query_out_names": list(out_names),
        "base_id": sample["base_id"],
        "variant": variant,
    }


def collate(records: list[dict], device) -> BatchGraph:
    fams = {r["family"] for r in records}
    if len(fams) != 1:
        raise ValueError("batch must share one structure family")
    fam = fams.pop()
    lay = family_layout(fam)
    B = len(records)
    n_der = len(lay["derived"])
    P = max(len(d[1]) for d in lay["derived"])
    ops = torch.zeros(B, n_der, dtype=torch.long)
    parents = torch.zeros(B, n_der, P, dtype=torch.long)
    roles = torch.zeros(B, n_der, P, dtype=torch.long)
    for j, (op, ps, rs) in enumerate(lay["derived"]):
        ops[:, j] = op
        parents[:, j, :len(ps)] = torch.tensor(ps)
        roles[:, j, :len(rs)] = torch.tensor(rs)
    labels = torch.tensor([r["state_labels"] for r in records], dtype=torch.long)
    n_out = max(len(r["query_out_idx"]) for r in records)
    qidx = torch.zeros(B, n_out, dtype=torch.long)
    for b, r in enumerate(records):
        qidx[b, :len(r["query_out_idx"])] = torch.tensor(r["query_out_idx"])
    return BatchGraph(
        input_texts=[r["input_texts"] for r in records],
        derived_ops=ops.to(device),
        derived_parents=parents.to(device),
        derived_roles=roles.to(device),
        node_types=[["num"] * len(lay["names"])] * B,
        state_labels=labels,
        query_texts=[r["query_text"] for r in records],
        answer_texts=[r["answer_text"] for r in records],
        query_out_idx=qidx.to(device),
        query_out_names=[r.get("query_out_names", []) for r in records],
    )


def load_records(jsonl_path: str, variants: tuple[str, ...], seed: int,
                 limit: int | None = None) -> list[dict]:
    import json
    rng = random.Random(seed)
    out = []
    with open(jsonl_path, encoding="utf-8") as f:
        for line in f:
            s = json.loads(line)
            for v in variants:
                out.append(sample_to_record(s, v, rng))
            if limit and len(out) >= limit:
                break
    rng.shuffle(out)
    return out[:limit] if limit else out


def group_by_family(records: list[dict]) -> dict[str, list[dict]]:
    out: dict[str, list[dict]] = {}
    for r in records:
        out.setdefault(r["family"], []).append(r)
    return out
