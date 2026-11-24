"""Revise conclusions by updating rules and facts in a belief graph."""
from __future__ import annotations

import os
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
import argparse
import json
import re


def load_belief():
    from datasets import load_dataset
    return load_dataset("CAiRE/belief_r")["test"]


def parse_premises(questions):
    """Parse premises into antecedent-consequent pairs and facts."""
    rules, facts = [], []
    for line in questions.split("\n"):
        line = line.strip()
        if not line or line.startswith("What") or line.startswith("("):
            continue
        m = re.match(r"^If (.+?), then (.+)$", line)
        if m:
            rules.append((m.group(1).strip(), m.group(2).strip()))
        else:
            facts.append(line)
    return rules, facts


def normalize(s):
    return re.sub(r"\s+", " ", s.lower().strip().rstrip("."))


def strip_neg(s):
    """Return the affirmative form used to match consequents."""
    s = normalize(s)
    s = re.sub(r"\b(does not|do not|did not|is not|are not|was not|were not|"
               r"will not|would not|can not|cannot|could not|should not|"
               r"has not|have not|had not|never|not|no)\b", "", s)
    return re.sub(r"\s+", " ", s).strip()


def stem_match(a, b):
    """Match inflectional variants such as feel and feels."""
    def stems(s):
        out = []
        for w in normalize(s).split():
            w = re.sub(r"(ies)$", "y", w)
            w = re.sub(r"(es|s)$", "", w)
            out.append(w)
        return set(out)
    sa, sb = stems(a), stems(b)
    if not sa or not sb:
        return False
    # 词集合高度重叠即匹配
    return len(sa & sb) >= max(1, min(len(sa), len(sb)) - 1)


def solve_belief(rules, facts):
    """Return the positive, negative or uncertain label for a ponens or tollens conclusion."""
    # 找主规则与查询结论：结论选项是关于某命题 P 的肯定/否定/不确定
    # 事实里含否定（does not / did not / is not / will not）
    neg_facts = [f for f in facts if re.search(r"\b(not|never|no)\b", f, re.I)]
    pos_facts = [f for f in facts if f not in neg_facts]

    # 匹配事实到规则后件/前件
    # tollens: 非Q + (P->Q) ⇒ 非P
    # 若存在替代规则 R->Q（Q有别的因），则非Q不能推出非P ⇒ 不确定
    for (ant, cons) in rules:
        cn = normalize(cons)
        # 事实否定后件？（去掉否定词后匹配后件）
        for nf in neg_facts:
            if cn and stem_match(strip_neg(nf), cn):
                # tollens 触发：非Q。检查是否有替代规则也推出 Q
                alt = [r for r in rules if stem_match(r[1], cons) and not stem_match(r[0], ant)]
                if alt:
                    return "c"   # 有替代路径 → 不确定（C0 更新）
                return "b"       # 无替代 → 非P（确定否定）
        # ponens: P + (P->Q) ⇒ Q
        for pf in pos_facts:
            if stem_match(ant, pf):
                return "a"
    return "c"


def belief_revision(ex_t, ex_t1):
    """Construct the initial belief graph and revise it after the new premise."""
    rules0, facts0 = parse_premises(ex_t["questions"])
    a0 = solve_belief(rules0, facts0)
    rules1, facts1 = parse_premises(ex_t1["questions"])
    # 编辑：新增的规则/事实
    new_rules = [r for r in rules1 if r not in rules0]
    # C1：新规则相关 ⟺ 其前件与原规则前件指同一事件（词干重叠高）
    # BU: 新前件 "John plays chess to win a game" ≈ 原前件 "John plays chess"（同事件）
    # BM: 新前件 "Jessica is invested..." ≠ 原前件 "John plays chess"（不同事件）
    orig_ants = [normalize(a) for (a, _) in rules0]
    def antecedent_overlap(new_ant):
        na = set(normalize(new_ant).split())
        best = 0.0
        for oa in orig_ants:
            oas = set(oa.split())
            if not oas:
                continue
            inter = len(na & oas)
            union = len(na | oas)
            best = max(best, inter / union if union else 0)
        return best
    relevant = [r for r in new_rules if antecedent_overlap(r[0]) >= 0.4]
    if relevant:
        # C0/C3：相关编辑 → 局部重算结论
        a1 = solve_belief(rules1, facts1)
    else:
        # C1：无关编辑 → 保持原结论
        a1 = a0
    return a0, a1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    data = load_belief()
    by_id = {}
    for ex in data:
        by_id.setdefault(ex["dataset_id"], {})[ex["step"]] = ex
    pairs = [v for v in by_id.values() if "time_t" in v and "time_t1" in v]
    if args.limit:
        pairs = pairs[:args.limit]
    print(f"pairs: {len(pairs)}", flush=True)

    bu_ok = bu_n = bm_ok = bm_n = t_ok = t_n = 0
    rows = []
    for pr in pairs:
        tt, tt1 = pr["time_t"], pr["time_t1"]
        a0, a1 = belief_revision(tt, tt1)
        gt0, gt1 = tt["ground_truth"], tt1["ground_truth"]
        t_ok += int(a0 == gt0); t_n += 1
        if gt1 != gt0:
            bu_ok += int(a1 == gt1); bu_n += 1
        else:
            bm_ok += int(a1 == gt1); bm_n += 1
        rows.append({"id": tt["dataset_id"], "gt0": gt0, "a0": a0,
                     "gt1": gt1, "a1": a1})
    bu = bu_ok / max(bu_n, 1); bm = bm_ok / max(bm_n, 1)
    summary = {"method": "belief_graph_edit_C0C1C3", "n_pairs": len(pairs),
               "time_t_acc": t_ok / max(t_n, 1),
               "BU_Acc": bu, "BU_n": bu_n, "BM_Acc": bm, "BM_n": bm_n,
               "BREU": (bu + bm) / 2}
    with open(os.path.join(args.out, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    with open(os.path.join(args.out, "rows.jsonl"), "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
