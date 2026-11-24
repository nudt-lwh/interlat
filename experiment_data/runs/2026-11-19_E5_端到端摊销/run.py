from pathlib import Path
import sys
folder = Path(__file__).resolve().parent
sys.dont_write_bytecode = True
sys.path.insert(0, str(folder.parent / "2026-11-19_P0-2"))
"""Measure serial online edits on the same graph. Run cached then cuda/cpu, then summarize."""
from shared import *
from time import perf_counter_ns
from statistics import median
from math import isclose
marks=(1,10,100,1000,5000)
r=next(r for r in problems() if r["width"]==12 and r["mixed"] and identity(r)["range"]=="digits" and r["task"]==0)
original=r["inputs"]
graph=r["graph"]
model=frozen_model()
rng=random.Random(551119)
script=[]
v=original.copy()
for _ in range(max(marks)):
    t=rng.randrange(1,len(v))
    v[t]=change(rng,v[t],(0,9))
    script.append((t,v[t],build(v,graph,truth)[-1]))

def update(state,t,value):
    z=state.copy()
    z[t]=value
    dirty={t}
    for i,(op,l,r) in enumerate(graph,len(original)):
        if l in dirty or r in dirty:
            z[i]=model.execute(op,z[l],z[r])
            dirty.add(i)
    return z

def bench(device,limit):
    if device=="cached":
        def encode(vals):return [model.encode("s",v) for v in vals]
    else:
        sys.path.insert(0,str(folder.parent/"2026-11-19_P0-1"))
        from encoder import FrozenEncoder,torch
        torch.set_num_threads(4)
        enc=FrozenEncoder(device)
        def encode(vals):
            # All literals are single digits; one true encoder forward per call.
            # features are transferred to CPU in __call__, synchronizing the device.
            return (enc([f"s = {v}" for v in vals])-model.mean)@model.weights + model.bias
    def initial(vals):
        return build(encode(vals),graph,model.execute)
    def single(vals,target,value,state,mode):
        if mode=="local":
            z=update(state,target,encode([value])[0])
        else:
            z=initial(vals)
        return z,round(z[-1])
    # Warm model, tokenizer and kernels at each actual batch size. Model startup excluded.
    for j in range(3):
        tmp=initial(original)
        update(tmp,1,encode([original[1]])[0])
    output=folder/(device+".jsonl")
    reps=5 if device=="cached" else 1
    if output.exists():
        raise FileExistsError(output)
    with output.open("x") as f:
        for rep in range(reps):
            vals=original.copy()
            start=perf_counter_ns()
            state=initial(vals)
            init_ns=perf_counter_ns()-start
            cum={"local":init_ns,"rebuild":0}
            for j,(t,value,gold) in enumerate(script[:limit],1):
                vals[t]=value
                order=("local","rebuild") if (j+rep)%2 else ("rebuild","local")
                durations={}
                predictions={}
                for mode in order:
                    start=perf_counter_ns()
                    z,pred=single(vals,t,value,state,mode)
                    ns=perf_counter_ns()-start
                    if mode=="local":state=z
                    durations[mode]=ns
                    cum[mode]+=ns
                    predictions[mode]=pred
                record(f,dict(device=device,repeat=rep,step=j,init_ns=init_ns,target=t,value=value,truth=gold,
                              pred=predictions,elapsed_ns=durations,cumulative_ns=cum))
                if j in marks or j%100==0:
                    f.flush()
                    if j in marks:print(device,rep,j,cum,flush=True)

def summarize():
    summary={}
    with (folder/"rows.jsonl").open("w") as out:
        for dev in ("cached","cpu","cuda"):
            p=folder/(dev+".jsonl")
            if not p.exists(): continue
            rows=[json.loads(line) for line in p.open()]
            for x in rows:record(out,x)
            bystep=defaultdict(list)
            for x in rows:bystep[x["step"]].append(x)
            report={}
            crossover=None
            for step,xs in sorted(bystep.items()):
                loc,reb=[median(x["cumulative_ns"][k] for x in xs)/1e6 for k in ("local","rebuild")]
                if loc<=reb and crossover is None:crossover=step
                if step in marks:
                    report[step]=dict(local_ms=loc,rebuild_ms=reb,local_ms_per_edit=loc/step,
                                      rebuild_ms_per_edit=reb/step,speedup=reb/loc)
            summary[dev]=dict(checkpoints=report,first_observed_break_even=crossover,rows=len(rows),
                initial_ms=median(x["init_ns"] for x in bystep[1])/1e6,
                max_step=max(bystep),correct={k:sum(x["pred"][k]==x["truth"] for x in rows) for k in ("local","rebuild")})
    save(folder,"summary",summary)
    import torch, transformers, winreg
    with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE,r"HARDWARE\DESCRIPTION\System\CentralProcessor\0") as key:
        cpu_name=winreg.QueryValueEx(key,"ProcessorNameString")[0].strip()
    artifacts(folder, cpu=cpu_name,gpu=torch.cuda.get_device_name(0) if "cuda" in summary else None,
       cpu_torch_threads=4,torch_version=torch.__version__,transformers_version=transformers.__version__,
       input_encoder="Qwen3-1.7B plus archived fork_best.pt encoder adapter, CPU bf16 embeddings, FP32 transformer blocks and projection; canonical template s = digit",source_task=identity(r),graph=graph,inputs=original,edit_seed=551119,private_only=True,
       checkpoints=marks, modes=list(summary),cached_repeats=5,uncached_repeats=1,processor=platform.processor(),
       design="single fixed mixed 12-branch graph; online private single-digit edits, alternating timing order per step. Cache path starts with existing canonical digit features and warmed decoder. Uncached local pays initial all-input batch plus one actual Qwen forward per changed literal; rebuild pays one 13-literal forward batch per edit. Both include graph computation and readout. Same model already loaded; no cross-step batching. Input array mutation, stream generation and scoring excluded identically. CPU cached median of repeated cumulative times; first observed crossing uses every measured prefix.")
if __name__=="__main__":
    if sys.argv[1]=="summarize":summarize()
    else:bench(sys.argv[1],int(sys.argv[2]) if len(sys.argv)>2 else 5000)
