"""Check the frozen gate, archived rule alignment and paired BREU difference."""
import json
from pathlib import Path
import numpy as np

here=Path(__file__).resolve().parent
rows=[json.loads(s) for s in (here/"rows.jsonl").read_text().splitlines()]
cfg=json.loads((here/"config.json").read_text())
stats=json.loads((here/"summary.json").read_text())
archive=here.parent/"belief_graph_v4"/"rows.jsonl"
original={r["id"]:r for r in (json.loads(s) for s in archive.read_text().splitlines())}
assert len(rows)==len({r["id"] for r in rows})==641
diff=[[],[]]
counts={k:[0,0] for k in ("fused","rule","copy","neural")}
for r in rows:
    a=original[r["id"]]
    assert tuple("abc".index(a[k]) for k in ("gt0","gt1","a1"))==(r["gt0"],r["gt1"],r["rule"])
    route=r["neural"]!=r["copy"] and r["confidence"]>=cfg["threshold"]
    assert r["used_neural"]==route and r["fused"]==(r["neural"] if route else r["rule"])
    i=int(r["gt0"]==r["gt1"])
    for k in counts:
        counts[k][i]+=int(r[k]==r["gt1"])
    diff[i].append(int(r["fused"]==r["gt1"])-int(r["rule"]==r["gt1"]))
assert [len(d) for d in diff]==[509,132]
for k,(u,m) in counts.items():
    assert abs((u/509+m/132)/2-stats[k]["breu"])<1e-12
rng=np.random.default_rng(45290)
draws=sum(rng.choice(d,(10000,len(d))).mean(1) for d in diff)/2
result=dict(correct_update_maintain=counts,paired_fusion_minus_rule=float(np.mean([np.mean(d) for d in diff])),
            paired_stratified_ci95=np.quantile(draws,[0.025,0.975]).tolist(),bootstrap_seed=45290,replicates=10000,
            fixed_joint_update=sum(r["copy"]==r["gt0"] and r["neural"]==r["gt1"] and r["gt0"]!=r["gt1"] for r in rows))
(here/"verified.json").write_text(json.dumps(result,indent=2))
print(result)
