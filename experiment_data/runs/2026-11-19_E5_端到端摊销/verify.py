from pathlib import Path
import sys
folder = Path(__file__).resolve().parent
sys.dont_write_bytecode = True
sys.path.insert(0, str(folder.parent / "2026-11-19_P0-2"))
import json
from collections import defaultdict
from statistics import median
from math import isclose
summary=json.loads((folder/"summary.json").read_text())
g=defaultdict(list)
with (folder/"rows.jsonl").open() as f:
    for line in f:
        x=json.loads(line)
        g[x["device"],x["repeat"]].append(x)
for (d,rep),rr in g.items():
    totals={"local":rr[0]["init_ns"],"rebuild":0}
    for j,x in enumerate(rr,1):
        assert x["step"]==j and x["init_ns"]==rr[0]["init_ns"]
        for k in totals:
            totals[k]+=x["elapsed_ns"][k]
            assert totals[k]==x["cumulative_ns"][k]
    assert len(rr)==summary[d]["max_step"]
reference=[(x["target"],x["value"],x["truth"]) for x in g["cached",0]]
for rr in g.values():
    assert reference[:len(rr)]==[(x["target"],x["value"],x["truth"]) for x in rr]
for d,obj in summary.items():
    reps=[rr for (device,rep),rr in g.items() if d==device]
    assert obj["rows"]==sum(map(len,reps))
    for k in ("local","rebuild"):
        assert obj["correct"][k]==sum(x["pred"][k]==x["truth"] for rr in reps for x in rr)
    expected=None
    for j,xx in enumerate(zip(*reps),1):
        loc,reb=[median(x["cumulative_ns"][k] for x in xx)/1e6 for k in ("local","rebuild")]
        if expected is None and loc<=reb:expected=j
        if str(j) in obj["checkpoints"]:
            z=obj["checkpoints"][str(j)]
            assert z["local_ms"]==loc and z["rebuild_ms"]==reb and isclose(z["speedup"],reb/loc)
    assert obj["first_observed_break_even"]==expected
assert summary["cached"]["max_step"]==summary["cuda"]["max_step"]==5000
print("Verified all raw intervals, cumulative times, paired input streams, accuracy, ratios, and crossovers.")
