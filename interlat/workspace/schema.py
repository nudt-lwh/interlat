"""Define the node, query and dependency structures of a workspace."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional


# 节点类型（首版）
TYPES = ("count", "currency", "bool", "set")

# 算子类别（首版任务族A）
OPERATORS = (
    "input", "add", "sub", "mul", "floordiv",
    "gt", "lt", "min", "max", "select",
)

# 入边角色：保证减法/整除/条件选择的操作数语义不随节点重排改变
ARG_ROLES = {
    "add": ("lhs", "rhs"),
    "sub": ("lhs", "rhs"),
    "mul": ("lhs", "rhs"),
    "floordiv": ("num", "den"),
    "gt": ("lhs", "rhs"),
    "lt": ("lhs", "rhs"),
    "min": ("lhs", "rhs"),
    "max": ("lhs", "rhs"),
    "select": ("condition", "then", "else"),
}


@dataclass
class Node:
    node_id: str                 # 实体与变量的唯一标识（训练时随机重命名）
    type: str                    # count/currency/bool/set
    operator: str                # input 或 OPERATORS 中的计算算子
    parents: tuple[str, ...] = ()
    arg_roles: tuple[str, ...] = ()
    value: Any = None            # 符号版真值；神经版此字段仅用于训练监督/评分
    z: Any = None                # 连续状态（神经版填充，首版设计 2×256 维）
    version: int = 0
    stale: bool = False
    local_fact: str = ""         # 输入节点的局部事实描述（编码用，不含其他输入）

    def __post_init__(self) -> None:
        if self.type not in TYPES:
            raise ValueError(f"unknown type: {self.type}")
        if self.operator not in OPERATORS:
            raise ValueError(f"unknown operator: {self.operator}")
        if self.operator == "input":
            if self.parents:
                raise ValueError("input node must have no parents")
        else:
            expected = ARG_ROLES[self.operator]
            if len(self.parents) != len(expected):
                raise ValueError(
                    f"{self.operator} expects {len(expected)} parents, got {len(self.parents)}"
                )
            if self.arg_roles != expected:
                raise ValueError(
                    f"{self.operator} arg_roles must be {expected}, got {self.arg_roles}"
                )


@dataclass
class Workspace:
    nodes: dict[str, Node] = field(default_factory=dict)
    queries: dict[str, tuple[str, ...]] = field(default_factory=dict)  # query→输出节点集合

    def add(self, node: Node) -> None:
        if node.node_id in self.nodes:
            raise ValueError(f"duplicate node: {node.node_id}")
        for p in node.parents:
            if p not in self.nodes:
                raise ValueError(f"parent {p} not added before {node.node_id}")
        self.nodes[node.node_id] = node

    def topo_order(self) -> list[str]:
        """Return a Kahn topological ordering of the dependency graph."""
        indeg = {nid: len(n.parents) for nid, n in self.nodes.items()}
        children: dict[str, list[str]] = {nid: [] for nid in self.nodes}
        for nid, n in self.nodes.items():
            for p in n.parents:
                children[p].append(nid)
        ready = sorted(nid for nid, d in indeg.items() if d == 0)
        order: list[str] = []
        while ready:
            cur = ready.pop(0)
            order.append(cur)
            for c in children[cur]:
                indeg[c] -= 1
                if indeg[c] == 0:
                    ready.append(c)
                    ready.sort()
        if len(order) != len(self.nodes):
            raise ValueError("cycle detected in workspace graph")
        return order

    def descendants(self, roots: set[str]) -> set[str]:
        children: dict[str, list[str]] = {nid: [] for nid in self.nodes}
        for nid, n in self.nodes.items():
            for p in n.parents:
                children[p].append(nid)
        seen: set[str] = set()
        stack = list(roots)
        while stack:
            cur = stack.pop()
            for c in children[cur]:
                if c not in seen:
                    seen.add(c)
                    stack.append(c)
        return seen

    def ancestors(self, targets: set[str]) -> set[str]:
        seen: set[str] = set()
        stack = list(targets)
        while stack:
            cur = stack.pop()
            for p in self.nodes[cur].parents:
                if p not in seen:
                    seen.add(p)
                    stack.append(p)
        return seen | set(targets)

    def affected(self, edited: set[str], query: Optional[str] = None) -> set[str]:
        """Intersect descendants of edited nodes with ancestors of the query, if supplied."""
        desc = self.descendants(edited)
        if query is None:
            return desc
        return desc & self.ancestors(set(self.queries[query]))
