from pathlib import Path
import sys
folder = Path(__file__).resolve().parent
sys.dont_write_bytecode = True
sys.path.insert(0, str(folder.parent / "2026-11-19_P0-2"))
from shared import *
from time import perf_counter_ns
from statistics import median

model = frozen_model()
seed, repeats = 13802,5
artifacts(folder, design="5 counterbalanced warm CPU repeats per graph and path, median per graph; complete state construction with cached canonical digit features; branch 1, branch 2, root; cumulative medians summed across graphs",
          repeats=repeats,order_seed=seed,processor=platform.processor(),timer="perf_counter_ns")
rows = problems()
random.Random(seed).shuffle(rows)
def answer(vals, graph, nodes, reuse):
    if reuse:
        state = rebuild(model, vals, graph)
        return [round(state[n]) for n in nodes]
    return [round(rebuild(model, vals, graph)[n]) for n in nodes]

for i in range(40):
    r = rows[i]
    nodes = (r["width"]+1,r["width"]+2,-1)
    for mode in (True,False):
        answer(r["inputs"],r["graph"],nodes,mode)
totals = [0,0]
counts = [[0]*3 for _ in range(2)]
with (folder/"rows.jsonl").open("w") as f:
    for i,r in enumerate(rows):
        vals,graph = r["inputs"],r["graph"]
        nodes = [r["width"]+1,r["width"]+2,len(vals)+len(graph)-1]
        ref = build(vals,graph,truth)
        gold = [ref[n] for n in nodes]
        samples = [[],[]]
        answers = [None,None]
        for repeat in range(repeats):
            first = (repeat+i)%2
            for mode in (first,1-first):
                start = perf_counter_ns()
                answers[mode] = answer(vals,graph,nodes,mode==0)
                samples[mode].append(perf_counter_ns()-start)
        for k in range(2):
            totals[k] += median(samples[k])
            for j in range(3):
                counts[k][j] += answers[k][j]==gold[j]
        record(f,dict(**identity(r),nodes=nodes,gold=gold,reuse=answers[0],rebuild=answers[1],
                      reuse_ns=samples[0],rebuild_ns=samples[1]))
save(folder,"summary",dict(graphs=len(rows),queries_per_graph=3,repeats=repeats,
    total_reuse_ms=totals[0]/1e6,total_rebuild_ms=totals[1]/1e6,total_speedup=totals[1]/totals[0],
    reuse_correct_by_query=counts[0],rebuild_correct_by_query=counts[1]))