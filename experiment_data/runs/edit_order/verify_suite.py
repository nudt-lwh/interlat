from pathlib import Path
import sys
folder = Path(__file__).resolve().parent
sys.dont_write_bytecode = True
sys.path.insert(0, str(folder.parent / "edit_order"))
"""Independent row-to-summary audit for the five CPU runs; run after all five run.py scripts."""
import json
from collections import defaultdict
from statistics import mean,median,stdev
from fractions import Fraction
from math import isclose

def read(k):
    path = folder.parent / k
    with (path/"rows.jsonl").open() as f:
        rows = [json.loads(line) for line in f]
    return rows,json.loads((path/"summary.json").read_text())

def taskkey(r):
    return (r["range"],r["width"],r["mixed"],r["task"])
rs,s = read("edit_order")
orders = defaultdict(dict)
for r in rs:
    orders[taskkey(r)][r["order"]] = r["steps"]
assert len(orders) == 2000 and len(rs) == 10000
for name,z in s["orders"].items():
    paths = [v[name] for v in orders.values()]
    assert z["paths"] == len(paths)
    assert z["final_correct"] == sum(p[-1]["truth"]==p[-1]["pred"] for p in paths)
    steps=[e for p in paths for e in p]
    assert z["steps"]==len(steps)
    assert z["preserved"]==sum(e["preserved"] for e in steps)
    assert z["agreement"]==sum(e["rebuild"]==e["pred"] for e in steps)
    assert z["mean_executed_by_step"]==[mean(p[i]["executed"] for p in paths) for i in range(len(paths[0]))]
for v in orders.values():
    for a,b in [("A_B","B_A"),("A_shared","shared_A")]:
        assert [(e["target"],e["new"]) for e in v[a]]==[(e["target"],e["new"]) for e in v[b][::-1]]
        assert v[a][-1]["pred"]==v[b][-1]["pred"]
rs,s=read("edit_sequences")
assert len(rs)==10000 and len({(taskkey(r),r["step"]) for r in rs})==10000
for j,z in s["steps"].items():
    g=[r for r in rs if r["step"]==int(j)]
    assert len(g)==z["n"]==500 and z["correct"]==sum(r["pred"]==r["truth"] for r in g)
    assert z["preserved"]==sum(r["preserved"] for r in g)
    assert z["agreement"]==sum(r["pred"]==r["rebuild"] for r in g)
    assert z["mean_cumulative_executed"]==mean(r["cumulative_executed"] for r in g)
rs,s=read("query_reuse")
assert len(rs)==len({taskkey(r) for r in rs})==s["graphs"]
for k in ("reuse","rebuild"):
    assert s[k+"_correct_by_query"]==[sum(r[k][i]==r["gold"][i] for r in rs) for i in range(3)]
    assert s["total_"+k+"_ms"]==sum(median(r[k+"_ns"]) for r in rs)/1e6
assert isclose(s["total_speedup"],s["total_rebuild_ms"]/s["total_reuse_ms"])
rs,s=read("evaluation_seeds")
assert len(rs)==32400
for seed,z in s["seeds"].items():
    g=[r for r in rs if r["data_seed"]==int(seed)]
    assert len(g)==len({(taskkey(r),r["target"]) for r in g})==z["n"]==10800
    for k in ("agreement","correct","preserved"):
        expected = sum((r["pred"]==r["rebuild"] if k=="agreement" else r["pred"]==r["truth"] if k=="correct" else r["preserved"]) for r in g)
        assert z[k]==expected
for k,z in s["metrics"].items():
    rates=[v[k]/v["n"] for v in s["seeds"].values()]
    assert z==dict(mean=mean(rates),sample_sd=stdev(rates))

rs,s=read("extended_operators")
assert len(rs)==len({(r["range"],r["width"],r["reverse"],r["task"],r["target"]) for r in rs})==10800
predicates = {0:lambda l,r:l+r,1:lambda l,r:l-r,2:lambda l,r:l*r,3:lambda l,r:Fraction(l,r),4:lambda l,r:int(l>r),5:lambda l,r:int(l<r)}
for r in rs:
    vals=r["inputs"].copy()
    vals[r["target"]]=r["new"]
    for op,l,rr in r["graph"]:
        vals.append(predicates[op](vals[l],vals[rr]))
    assert vals[-1]==r["truth"]==r["symbolic"]
    for x in r["node_diagnostics"]:
        assert vals[x["node"]]==Fraction(x["truth"])
for name,rr in [("overall",rs)]+[(str(w),[r for r in rs if r["width"]==w]) for w in (5,7,12)]:
    z = s["overall"] if name=="overall" else s["by_width"][name]
    assert z["n"]==len(rr)
    for k,field in [("correct","pred"),("symbolic_correct","symbolic")]:
        assert z[k]==sum(r["truth"]==r[field] for r in rr)
    assert z["agreement"]==sum(r["pred"]==r["rebuild"] for r in rr)
    assert z["preserved"]==sum(r["preserved"] for r in rr)
print("PASS: audited 10,000 + 10,000 + 5,400 + 32,400 + 10,800 raw rows; reproduced P2-2 oracle with independent Fraction evaluator.")