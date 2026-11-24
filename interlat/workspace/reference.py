"""符号参考器：只用于生成训练目标与评分真值。

推理时神经 Execute 接收父节点连续状态，不调用本模块（方案 v1.0 明确：
否则退化为普通符号程序执行）。
"""
from __future__ import annotations

from typing import Any

from .schema import Workspace


def compute(op: str, vals: tuple[Any, ...]) -> Any:
    """vals 已按 arg_role 顺序排列。整除采用向下取整，除数不得为0。"""
    if op == "add":
        return vals[0] + vals[1]
    if op == "sub":
        return vals[0] - vals[1]
    if op == "mul":
        return vals[0] * vals[1]
    if op == "floordiv":
        if vals[1] == 0:
            raise ZeroDivisionError("floordiv denominator is 0")
        return vals[0] // vals[1]
    if op == "gt":
        return vals[0] > vals[1]
    if op == "lt":
        return vals[0] < vals[1]
    if op == "min":
        return min(vals[0], vals[1])
    if op == "max":
        return max(vals[0], vals[1])
    if op == "select":
        return vals[1] if vals[0] else vals[2]  # condition, then, else
    raise ValueError(f"cannot compute operator: {op}")


def evaluate(ws: Workspace) -> dict[str, Any]:
    """按拓扑序计算全部节点真值，返回 node_id→value。"""
    values: dict[str, Any] = {}
    for nid in ws.topo_order():
        node = ws.nodes[nid]
        if node.operator == "input":
            values[nid] = node.value
        else:
            vals = tuple(values[p] for p in node.parents)
            values[nid] = compute(node.operator, vals)
    return values


def check_range(values: dict[str, Any], bound: int = 999) -> bool:
    """首版数值范围约束：所有数值节点 |v| ≤ bound（布尔除外）。"""
    for v in values.values():
        if isinstance(v, bool):
            continue
        if not isinstance(v, int) or abs(v) > bound:
            return False
    return True
