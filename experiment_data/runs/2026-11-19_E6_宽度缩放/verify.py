from pathlib import Path
import sys
folder = Path(__file__).resolve().parent
sys.dont_write_bytecode = True
sys.path.insert(0, str(folder.parent / "2026-11-19_P0-2"))
from shared import *
from statistics import median
from math import isclose
s=json.loads((folder/"summary.json").read_text())["by_width"]
rs=[json.loads(line) for line in (folder/"rows.jsonl").open()]
assert len(rs)==len(set((r["width"],r["task"]) for r in rs))==900
for w,v in s.items():
    rr=[r for r in rs if r["width"]==int(w)]
    assert len(rr)==v["n"]==300 and sum(r["mixed"] for r in rr)==150
    for r in rr:
        vals=r["inputs"].copy()
        vals[r["target"]]=r["new"]
        assert build(vals,r["graph"],truth)[-1]==r["gold"]
        dirty={r["target"]}
        for node,(_,l,rt) in enumerate(r["graph"],len(vals)):
            if l in dirty or rt in dirty:dirty.add(node)
        assert len(dirty)-1==r["executed"]
    assert v["correct"]==sum(r["pred"]==r["gold"] for r in rr)
    assert v["full_correct"]==sum(r["rebuild"]==r["gold"] for r in rr)
    assert v["executed_ratio"]==sum(r["executed"] for r in rr)/sum(r["total"] for r in rr)
    assert isclose(v["total_speedup"],sum(median(r["full_ns"]) for r in rr)/sum(median(r["local_ns"]) for r in rr))
print("Verified all 900 reference roots, exact dependency counts, and paired timing aggregates.")
