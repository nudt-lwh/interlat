from pathlib import Path
import sys
folder = Path(__file__).resolve().parent
sys.dont_write_bytecode = True
sys.path.insert(0, str(folder.parent / "2026-11-19_P0-2"))
from shared import *
from statistics import mean

artifacts(folder, design="500 stratified graphs, alternating private then shared; the same 20-step trajectory supplies prefixes 5/10/20",sample_seed=38001,edit_seed=38002)
model = frozen_model()
rng = random.Random(38002)
by_step = defaultdict(list)
with (folder/"rows.jsonl").open("w") as f:
    for r in design_sample(500,38001):
        vals,graph = r["inputs"],r["graph"]
        state = rebuild(model, vals, graph)
        count = 0
        for j in range(1,21):
            target = 0 if j%2==0 else rng.randrange(1,len(vals))
            new = change(rng,vals[target],ranges[identity(r)["range"]])
            state,vals,s = edit(model,state,vals,graph,target,new)
            count += s["executed"]
            s["cumulative_executed"] = count
            record(f, dict(**identity(r),step=j,**s))
            by_step[j].append(s)
report = {k:dict(n=len(ss),correct=sum(s["correct"] for s in ss), preserved=sum(s["preserved"] for s in ss),
                 agreement=sum(s["agreement"] for s in ss),mean_cumulative_executed=mean(s["cumulative_executed"] for s in ss)) for k,ss in by_step.items()}
save(folder,"summary",dict(steps=report, final={k:report[k] for k in (5,10,20)}))