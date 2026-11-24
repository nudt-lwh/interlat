from pathlib import Path
import sys
folder = Path(__file__).resolve().parent
sys.dont_write_bytecode = True
sys.path.insert(0, str(folder.parent / "2026-11-19_P0-2"))
from shared import *
from statistics import mean

artifacts(folder, design="stratified 2000 formal graphs, independent fixed replacements, private/shared reverse as sensitivity",
          sample_seed=28001, edit_seed=28002)
model = frozen_model()
rng = random.Random(28002)
groups = defaultdict(list)
invariance = [0, 0]
with (folder/"rows.jsonl").open("w") as f:
    for r in design_sample(2000, 28001):
        original, graph = r["inputs"], r["graph"]
        initial = rebuild(model, original, graph)
        a,b = rng.sample(range(1,len(original)), 2)
        bounds = ranges[identity(r)["range"]]
        replacements = {k:change(rng,original[k],bounds) for k in (a,b,0)}
        outcomes = {}
        for name, targets in (("A_B",(a,b)), ("B_A",(b,a)), ("A_shared",(a,0)), ("shared_A",(0,a)), ("shared",(0,))):
            state,vals = initial, original
            steps = []
            for t in targets:
                state,vals,trace = edit(model,state,vals,graph,t,replacements[t])
                steps.append(trace)
            row = dict(**identity(r), order=name, steps=steps)
            record(f,row)
            groups[name].append(steps)
            outcomes[name] = steps[-1]["pred"]
        invariance[0] += outcomes["A_B"]==outcomes["B_A"]
        invariance[1] += outcomes["A_shared"]==outcomes["shared_A"]
summary = {}
for name,paths in groups.items():
    steps = [s for p in paths for s in p]
    summary[name] = dict(paths=len(paths), final_correct=sum(p[-1]["correct"] for p in paths),
                         steps=len(steps),preserved=sum(s["preserved"] for s in steps),
                         agreement=sum(s["agreement"] for s in steps),
                         mean_executed_by_step=[mean(p[j]["executed"] for p in paths) for j in range(len(paths[0]))])
save(folder, "summary", dict(orders=summary, private_order_agree=invariance[0], mixed_order_agree=invariance[1], paired=2000))