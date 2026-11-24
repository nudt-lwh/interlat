from pathlib import Path
import sys
folder = Path(__file__).resolve().parent
sys.dont_write_bytecode = True
sys.path.insert(0, str(folder.parent / "2026-11-19_P0-2"))
from shared import *
from iterate import topology
from statistics import median,mean
from time import perf_counter_ns

model=frozen_model()
seed=620501
rng=random.Random(seed)
items=[]
for w in (20,50,100):
    for i in range(300):
        mixed=bool(i%2)
        graph=topology(w,mixed,rng)
        vals=[rng.randint(0,9) for _ in range(w+1)]
        t=rng.randrange(1,w+1)
        new=change(rng,vals[t],(0,9))
        items.append((w,i,mixed,graph,vals,t,new))
rng.shuffle(items)
artifacts(folder,widths=[20,50,100],graphs_per_width=300,balanced_families=["add","mixed"],digits=[0,9],seed=seed,
   timing="cached digit features, prebuilt state; both paths re-encode edited input only, copy state, scan graph, execute selected/all derived nodes and round root. Six counterbalanced repeats per paired graph after warmup; per-graph medians, then ratio of summed medians. Dirty-set scan overhead included.",
   uncertainty="percentile 95% interval from 2000 graph-paired resamples stratified by graph family within width, conditional on this host",
   bootstrap_seed=620502,processor=platform.processor())

def run(state,graph,t,w,new,local):
    z=state.copy()
    z[t]=model.encode(f"x{t-1}",new)
    if local:
        dirty={t}
        for k,(op,l,r) in enumerate(graph,w+1):
            if l in dirty or r in dirty:
                z[k]=model.execute(op,z[l],z[r])
                dirty.add(k)
        n=len(dirty)-1
    else:
        for k,(op,l,r) in enumerate(graph,w+1):
            z[k]=model.execute(op,z[l],z[r])
        n=len(graph)
    return round(z[-1]),n

groups=defaultdict(list)
with (folder/"rows.jsonl").open("w") as f:
    for ix,(w,i,mixed,graph,vals,t,new) in enumerate(items):
        state=build([model.encode("s" if not k else f"x{k-1}",v) for k,v in enumerate(vals)],graph,model.execute)
        edited=vals.copy()
        edited[t]=new
        gold=build(edited,graph,truth)[-1]
        ts=[[],[]]
        for rep in range(-1,6):
            first=(ix+rep)%2
            for path in (first,1-first):
                start=perf_counter_ns()
                pred,n=run(state,graph,t,w,new,path==0)
                ns=perf_counter_ns()-start
                if path==0:local_pred,executed=pred,n
                else:full_pred=pred
                if rep>=0:ts[path].append(ns)
        row=dict(width=w,task=i,mixed=mixed,inputs=vals,graph=graph,target=t,new=new,gold=gold,pred=local_pred,
                 rebuild=full_pred,executed=executed,total=len(graph),local_ns=ts[0],full_ns=ts[1])
        record(f,row)
        groups[w].append(row)
summary={}
boot=np.random.default_rng(620502)
for w,rs in sorted(groups.items()):
    t=np.array([[median(r["local_ns"]),median(r["full_ns"])] for r in rs])
    strata=[np.flatnonzero([r["mixed"]==m for r in rs]) for m in (False,True)]
    ix=np.concatenate([boot.choice(g,size=(2000,len(g)),replace=True) for g in strata],axis=1)
    estimates=t[ix].sum(axis=1)
    summary[w]=dict(n=len(rs),correct=sum(r["pred"]==r["gold"] for r in rs),full_correct=sum(r["rebuild"]==r["gold"] for r in rs),
          total_nodes=rs[0]["total"],mean_executed=mean([r["executed"] for r in rs]),
          executed_ratio=sum(r["executed"] for r in rs)/sum(r["total"] for r in rs),
          mean_local_us=float(t[:,0].mean()/1000),mean_full_us=float(t[:,1].mean()/1000),
          total_speedup=float(t[:,1].sum()/t[:,0].sum()),bootstrap_ci=np.quantile(estimates[:,1]/estimates[:,0],[.025,.975]).tolist())
save(folder,"summary",dict(by_width=summary))
