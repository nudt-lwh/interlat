import hashlib
import json
import operator
from collections import Counter
from pathlib import Path
from functools import lru_cache
import numpy as np
from iterate import Candidate

out=Path(__file__).resolve().parent
cfg=json.loads((out/"frozen.json").read_text())
results=json.loads((out/"formal.json").read_text())
op=(operator.add,operator.sub,operator.mul)
summary={}
item_signatures=set()
for seed in cfg["fit_seeds"]:
    rows=[json.loads(s) for s in (out/f"formal_rows_{seed}.jsonl").read_text().splitlines()]
    counts=Counter()
    sampled=hashlib.sha256()
    max_abs=0
    for r in rows:
        inputs=r["after"]; nodes=r["graph"]
        @lru_cache(None)
        def solve(j):
            if j<len(inputs):return inputs[j]
            typ,l,b=nodes[j-len(inputs)]
            return op[typ](solve(l),solve(b))
        root=solve(len(inputs)+len(nodes)-1)
        assert root==r["truth"]
        counts["correct"]+=root==r["pred"]
        counts["preserved"]+=r["preserved"]
        counts["agreement"]+=r["pred"]==r["rebuild"]
        counts["skips_correct"]+=r["actual"]<r["total"] and root==r["pred"]
        max_abs=max(max_abs,abs(root))
        sampled.update(json.dumps([inputs,nodes,r["target"]]).encode())
    counts["events"]=len(rows)
    summary[seed]=dict(counts,max_absolute_answer=max_abs)
    item_signatures.add(sampled.hexdigest())
assert len(item_signatures)==1
previous=out.parent/"2026-11-19_云实验"
model=Candidate(42,True,True)
paired={}
for line in (previous/"fork_rows.jsonl").read_text().splitlines():
    r=json.loads(line);w=r["width"]
    xs=[model.encode(n,v) for n,v in zip(model.names,r["after"])]
    level=[model.execute(0,xs[0],x) for x in xs[1:]]
    while len(level)>1: level=[model.execute(0,l,h) for l,h in zip(level[::2],level[1::2])]
    d=paired.setdefault(w,[0,0,0])
    d[0]+=int(r["local"])==int(r["truth"]);d[1]+=round(level[0])==int(r["truth"]);d[2]+=1
print(json.dumps(dict(formal=summary,prior_same_items=paired),indent=2))
(out/"verified.json").write_text(json.dumps(dict(formal=summary,prior_same_items=paired),indent=2))
