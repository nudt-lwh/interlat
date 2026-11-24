from pathlib import Path
import sys
folder = Path(__file__).resolve().parent
sys.dont_write_bytecode = True
sys.path.insert(0, str(folder.parent / "edit_order"))
from shared import *
from statistics import mean,stdev
seeds = [1940271,2940271,3940271]
artifacts(folder, design="original generator, six width/family cells times three input ranges; 300 graphs per cell, two edits per graph", data_seeds=seeds, fit_seed_fixed=True)
model = frozen_model()
by_seed = {}
with (folder/"rows.jsonl").open("w") as f:
    for seed in seeds:
        counts = dict(n=0,correct=0,preserved=0,agreement=0)
        for name,bounds in ranges.items():
            rows = evaluate(model,(5,7,12),300,seed+bounds[0],bounds)
            for r in rows:
                record(f,dict(data_seed=seed,range=name,**r))
                counts["n"] += 1
                for k in ("correct","preserved"):
                    counts[k] += int(r[k])
                counts["agreement"] += r["pred"]==r["rebuild"]
        by_seed[seed] = counts
metrics = {}
for k in ("correct","preserved","agreement"):
    rates = [v[k]/v["n"] for v in by_seed.values()]
    metrics[k] = dict(mean=mean(rates),sample_sd=stdev(rates))
save(folder,"summary",dict(seeds=by_seed,metrics=metrics))