"""Apply edits and recompute dependent nodes in a symbolic workspace."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Optional

from .schema import Workspace
from . import reference


@dataclass
class Edit:
    target: str
    op: str          # set / add / toggle
    value: Any = None

    def __post_init__(self) -> None:
        if self.op not in ("set", "add", "toggle"):
            raise ValueError(f"unsupported edit op: {self.op}")


def apply_edit(
    ws: Workspace,
    edit: Edit,
    executor: Optional[Callable[[str, tuple], Any]] = None,
) -> dict[str, Any]:
    """Update the workspace in place and return recomputation statistics."""
    executor = executor or reference.compute
    node = ws.nodes[edit.target]
    if node.operator != "input":
        raise ValueError("first version only edits input nodes")

    if edit.op == "set":
        new_value = edit.value
    elif edit.op == "add":
        if node.type not in ("count", "currency"):
            raise ValueError("add only for numeric nodes")
        new_value = node.value + edit.value
    else:  # toggle
        if node.type != "bool":
            raise ValueError("toggle only for bool nodes")
        new_value = not node.value

    affected = ws.descendants({edit.target})
    node.value = new_value
    node.version += 1
    for nid in affected:
        ws.nodes[nid].stale = True

    recomputed = 0
    for nid in ws.topo_order():
        if nid not in affected:
            continue
        n = ws.nodes[nid]
        vals = tuple(ws.nodes[p].value for p in n.parents)
        n.value = executor(n.operator, vals)
        n.version += 1
        n.stale = False
        recomputed += 1

    return {
        "target": edit.target,
        "op": edit.op,
        "recomputed_nodes": recomputed,
        "preserved_nodes": len(ws.nodes) - 1 - recomputed,
        "affected": sorted(affected),
    }


def snapshot(ws: Workspace) -> dict[str, tuple[Any, int]]:
    """Record each node's value and version for preservation checks."""
    return {nid: (n.value, n.version) for nid, n in ws.nodes.items()}
