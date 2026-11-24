"""Compute reference values for supervision and evaluation."""
from __future__ import annotations

from typing import Any

from .schema import Workspace


def compute(op: str, vals: tuple[Any, ...]) -> Any:
    """Evaluate an operator on role-ordered parents; integer division uses floor semantics."""
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
    """Return reference values for every node in topological order."""
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
    """Check the absolute bound on numeric node values, excluding booleans."""
    for v in values.values():
        if isinstance(v, bool):
            continue
        if not isinstance(v, int) or abs(v) > bound:
            return False
    return True
