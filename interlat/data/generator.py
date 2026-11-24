"""任务族A生成器：多步数量与资源计算（采购/库存/电量/时间安排）。

结构家族（structure_family）＝算子骨架模板；同一基础任务的原版与全部编辑版
共享 base_id 与家族标签，划分以家族为单位，防止泄漏。
实体名与节点顺序在样本内随机化，防止固定槽位记忆。
"""
from __future__ import annotations

import json
import random
from dataclasses import asdict
from typing import Any

from ..workspace.schema import ARG_ROLES, Node, Workspace
from ..workspace import reference
from ..workspace.edit import Edit, apply_edit

# 结构家族模板：每个模板定义输入角色与计算骨架。
# 输入名用语义角色（实体前缀在样本内随机生成），算子骨架决定家族。
FAMILIES: dict[str, dict[str, Any]] = {
    "purchase_budget": {
        "inputs": ["budget", "unit_price", "shipping", "stock", "target"],
        "derived": [
            ("need", "sub", ("target", "stock")),
            ("goods_cost", "mul", ("need", "unit_price")),
            ("total_cost", "add", ("goods_cost", "shipping")),
            ("remaining", "sub", ("budget", "total_cost")),
            ("feasible", "gt", ("budget", "total_cost")),
        ],
        "queries": {"can_purchase": ("feasible",), "budget_gap": ("remaining",)},
        "editable": ["budget", "unit_price", "shipping", "stock", "target"],
    },
    "inventory_refill": {
        "inputs": ["stock", "daily_use", "days", "safety", "capacity"],
        "derived": [
            ("consumed", "mul", ("daily_use", "days")),
            ("required", "add", ("consumed", "safety")),
            ("refill", "sub", ("required", "stock")),
            ("fits", "lt", ("refill", "capacity")),
        ],
        "queries": {"refill_amount": ("refill",), "fits_capacity": ("fits",)},
        "editable": ["stock", "daily_use", "days", "safety", "capacity"],
    },
    "power_schedule": {
        "inputs": ["battery", "device_draw", "hours", "reserve", "solar_gain"],
        "derived": [
            ("usage", "mul", ("device_draw", "hours")),
            ("net", "sub", ("usage", "solar_gain")),
            ("left", "sub", ("battery", "net")),
            ("safe", "gt", ("left", "reserve")),
        ],
        "queries": {"power_left": ("left",), "power_safe": ("safe",)},
        "editable": ["battery", "device_draw", "hours", "reserve", "solar_gain"],
    },
    "time_plan": {
        "inputs": ["deadline", "task_time", "n_tasks", "break_time", "start"],
        "derived": [
            ("work", "mul", ("task_time", "n_tasks")),
            ("total", "add", ("work", "break_time")),
            ("finish", "add", ("start", "total")),
            ("on_time", "lt", ("finish", "deadline")),
        ],
        "queries": {"finish_time": ("finish",), "meets_deadline": ("on_time",)},
        "editable": ["deadline", "task_time", "n_tasks", "break_time", "start"],
    },
}

ENTITY_PREFIXES = ["alpha", "bravo", "cedar", "delta", "ember", "fjord", "gale", "harbor"]

TEMPLATES = {
    "purchase_budget": (
        "{ent} 采购任务：预算 {budget} 元，单价 {unit_price} 元，运费 {shipping} 元，"
        "现有库存 {stock} 件，目标总数 {target} 件。"
    ),
    "inventory_refill": (
        "{ent} 库存任务：现有 {stock} 件，日耗 {daily_use} 件，计划 {days} 天，"
        "安全余量 {safety} 件，仓容上限 {capacity} 件。"
    ),
    "power_schedule": (
        "{ent} 用电任务：电量 {battery}，设备功率 {device_draw}，运行 {hours} 小时，"
        "保留电量 {reserve}，光伏补充 {solar_gain}。"
    ),
    "time_plan": (
        "{ent} 排程任务：截止时刻 {deadline}，单任务耗时 {task_time}，共 {n_tasks} 项，"
        "休息 {break_time}，起始时刻 {start}。"
    ),
}


def _sample_inputs(rng: random.Random, family: str) -> dict[str, int]:
    """采样输入并保证派生值在 0～999 且 floordiv 无零除（本族无 floordiv）。"""
    if family == "purchase_budget":
        stock = rng.randint(0, 20)
        target = stock + rng.randint(1, 20)
        unit_price = rng.randint(1, 60)
        shipping = rng.randint(0, 50)
        need = target - stock
        total = need * unit_price + shipping
        budget = total + rng.randint(-100, 300)
        budget = max(0, min(999, budget))
        return dict(budget=budget, unit_price=unit_price, shipping=shipping,
                    stock=stock, target=target)
    if family == "inventory_refill":
        daily_use = rng.randint(1, 30)
        days = rng.randint(1, 10)
        safety = rng.randint(0, 50)
        stock = rng.randint(0, 100)
        required = daily_use * days + safety
        refill = required - stock
        capacity = max(0, min(999, refill + rng.randint(-50, 200)))
        return dict(stock=stock, daily_use=daily_use, days=days,
                    safety=safety, capacity=capacity)
    if family == "power_schedule":
        device_draw = rng.randint(1, 40)
        hours = rng.randint(1, 10)
        solar_gain = rng.randint(0, 100)
        reserve = rng.randint(0, 80)
        usage = device_draw * hours
        net = usage - solar_gain
        battery = max(0, min(999, net + reserve + rng.randint(-80, 250)))
        return dict(battery=battery, device_draw=device_draw, hours=hours,
                    reserve=reserve, solar_gain=solar_gain)
    if family == "time_plan":
        task_time = rng.randint(1, 40)
        n_tasks = rng.randint(1, 10)
        break_time = rng.randint(0, 60)
        start = rng.randint(0, 200)
        total = task_time * n_tasks + break_time
        deadline = max(0, min(999, start + total + rng.randint(-60, 200)))
        return dict(deadline=deadline, task_time=task_time, n_tasks=n_tasks,
                    break_time=break_time, start=start)
    raise ValueError(f"unknown family: {family}")


def build_workspace(rng: random.Random, family: str, inputs: dict[str, int],
                    ent: str) -> Workspace:
    spec = FAMILIES[family]
    ws = Workspace()
    # 节点随机重排：输入节点先以随机顺序加入（计算节点按依赖序）
    input_names = list(spec["inputs"])
    rng.shuffle(input_names)
    for name in input_names:
        nid = f"{ent}.{name}"
        ws.add(Node(
            node_id=nid,
            type="currency" if family == "purchase_budget" and name in
                 ("budget", "unit_price", "shipping") else "count",
            operator="input",
            value=inputs[name],
            local_fact=f"{name}={inputs[name]}",
        ))
    for name, op, parents in spec["derived"]:
        nid = f"{ent}.{name}"
        pids = tuple(f"{ent}.{p}" for p in parents)
        out_type = "bool" if op in ("gt", "lt") else (
            "currency" if family == "purchase_budget" and name in
            ("goods_cost", "total_cost", "remaining") else "count")
        ws.add(Node(node_id=nid, type=out_type, operator=op,
                    parents=pids, arg_roles=ARG_ROLES[op]))
    ws.queries = {q: tuple(f"{ent}.{n}" for n in outs)
                  for q, outs in spec["queries"].items()}
    return ws


def _edit_value(rng: random.Random, family: str, name: str,
                current: int) -> tuple[str, int]:
    """生成保证派生值仍在范围内的编辑（set/add）。"""
    spec_inputs = FAMILIES[family]["inputs"]
    for _ in range(64):
        if rng.random() < 0.6:
            op = "set"
            lo, hi = (0, 999)
            if name in ("unit_price", "daily_use", "device_draw", "task_time"):
                lo, hi = (1, 60)
            elif name in ("days", "hours", "n_tasks"):
                lo, hi = (1, 10)
            elif name in ("stock",):
                lo, hi = (0, 100)
            elif name in ("target",):
                lo = current + 1
                hi = min(999, current + 30)
                if lo > hi:
                    continue
            val = rng.randint(lo, hi)
        else:
            op = "add"
            val = rng.randint(-30, 30)
        if op == "add" and current + val < 0:
            continue
        if name == "target" and (val if op == "set" else current + val) <= current:
            continue
        return op, val
    return "set", current  # 兜底：不变编辑（验证时会被过滤）


def make_sample(rng: random.Random, base_id: str) -> dict[str, Any]:
    family = rng.choice(list(FAMILIES))
    ent = rng.choice(ENTITY_PREFIXES) + str(rng.randint(10, 99))
    inputs = _sample_inputs(rng, family)
    ws = build_workspace(rng, family, inputs, ent)
    values = reference.evaluate(ws)
    if not reference.check_range(values):
        return make_sample(rng, base_id)  # 重采样（概率极低）

    task_text = TEMPLATES[family].format(ent=ent, **inputs)
    spec = FAMILIES[family]

    # 两个单编辑＋一个组合编辑（不同输入节点，保证独立可交换）
    editable = list(spec["editable"])
    rng.shuffle(editable)
    edits: list[dict[str, Any]] = []
    used: set[str] = set()
    for i in range(3):
        name = editable[i % len(editable)]
        while name in used and len(used) < len(editable):
            name = rng.choice(editable)
        used.add(name)
        op, val = _edit_value(rng, family, name, inputs[name])
        edits.append({"target": f"{ent}.{name}", "op": op, "value": val,
                      "kind": "single" if i < 2 else "combo_member"})

    # 参考结果：原版、每个单编辑后、组合（前两个单编辑顺序施加）后
    def run(edits_seq: list[dict[str, Any]]) -> dict[str, Any]:
        w = build_workspace(random.Random(0), family, inputs, ent)
        init_vals = reference.evaluate(w)
        for nid, v in init_vals.items():
            w.nodes[nid].value = v
        stats = []
        for e in edits_seq:
            stats.append(apply_edit(w, Edit(e["target"], e["op"], e["value"])))
        vals = {nid: n.value for nid, n in w.nodes.items()}
        return {"states": vals,
                "answers": {q: [vals[n] for n in outs]
                            for q, outs in w.queries.items()},
                "edit_stats": stats}

    single_runs = [run([e]) for e in edits[:2]]
    combo_forward = run(edits[:2])
    combo_reverse = run(list(reversed(edits[:2])))

    # 范围守卫：任一编辑路线的派生值越界则重采样，保证全部参考状态在 0～999
    for ref in single_runs + [combo_forward, combo_reverse]:
        if not reference.check_range(ref["states"]):
            return make_sample(rng, base_id)

    return {
        "base_id": base_id,
        "structure_family": family,
        "entity": ent,
        "task_text": task_text,
        "graph": {"nodes": [asdict(n) for n in ws.nodes.values()],
                  "queries": {q: list(o) for q, o in ws.queries.items()}},
        "initial_state": inputs,
        "edits": edits,
        "queries": list(ws.queries),
        "reference": {
            "original": run([]),
            "single": single_runs,
            "combo_forward": combo_forward,
            "combo_reverse": combo_reverse,
        },
        "split_group": family,
    }


def generate(n: int, seed: int) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    return [make_sample(rng, f"task_{i:06d}") for i in range(n)]


def split_by_family(samples: list[dict[str, Any]], seed: int,
                    dev_ratio: float = 0.2, test_ratio: float = 0.2
                    ) -> dict[str, list[dict[str, Any]]]:
    """按结构家族划分不可行（家族只有4个且都需覆盖），故按基础任务的
    实体+结构签名划分：同一家族内按 base_id 哈希分组，保证同一基础任务
    的所有变体同组。家族本身在 train/dev/test 都出现（家族级泛化由
    未见变量对组合测试承担，见方案 C2）。"""
    rng = random.Random(seed)
    ids = [s["base_id"] for s in samples]
    rng.shuffle(ids)
    n_dev = int(len(ids) * dev_ratio)
    n_test = int(len(ids) * test_ratio)
    dev_ids, test_ids = set(ids[:n_dev]), set(ids[n_dev:n_dev + n_test])
    out = {"train": [], "dev": [], "test": []}
    for s in samples:
        if s["base_id"] in dev_ids:
            out["dev"].append(s)
        elif s["base_id"] in test_ids:
            out["test"].append(s)
        else:
            out["train"].append(s)
    return out


def dump_jsonl(samples: list[dict[str, Any]], path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for s in samples:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")
