"""Paired uncached edit-forward study. python benchmark.py cuda; python benchmark.py cpu."""
import json
import sys
import re
import platform
from collections import defaultdict
from random import Random
from statistics import median
from time import perf_counter_ns
from pathlib import Path

folder=Path(__file__).resolve().parent
sys.path.insert(0,str(folder.parent/"2026-11-19_P0-1"))
from encoder import Candidate, FrozenEncoder, build, edit, formal_rows, number, torch, np

def cpu_name():
    if sys.platform == "win32":
        import winreg
        with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE,r"HARDWARE\DESCRIPTION\System\CentralProcessor\0") as key:
            return winreg.QueryValueEx(key,"ProcessorNameString")[0].strip()
    return platform.processor()

def sample():
    groups=defaultdict(list)
    for i,r in enumerate(formal_rows(),1):
        groups[r["width"],r["mixed"],len(str(r["after"][r["target"]])),r["target"]==0].append((i,r))
    rng=Random(70011)
    rows=[item for group in groups.values() for item in rng.sample(group,2)]
    rng.shuffle(rows)
    return rows

def run(device):
    torch.set_num_threads(4)
    model=Candidate(42,True,True)
    enc=FrozenEncoder(device)
    items=sample()
    rows=[]
    for i,(source_line,r) in enumerate(items[:4]+items):
        names=["s"]+[f"x{j}" for j in range(r["width"])]
        state=build([model.encode(n,v) for n,v in zip(names,r["inputs"])],r["graph"],model.execute)
        k=r["target"]
        text=f"{names[k]} = {r['after'][k]}"
        if device=="cuda": torch.cuda.synchronize()
        start=perf_counter_ns()
        match=re.fullmatch(r"(\w+) = ([0-9]+)",text)
        features=enc([f"s = {c}" for c in match[2]])
        value=number(model,(features-model.mean)@model.weights+model.bias)
        local,dirty=edit(model,state,k,value,r["graph"],len(names))
        pred=round(local[-1])
        if device=="cuda": torch.cuda.synchronize()
        elapsed=perf_counter_ns()-start
        if i>=4:
            rows.append(dict(device=device,source_line=source_line,width=r["width"],mixed=r["mixed"],
                        digits=len(match[2]),shared=k==0,text=text,truth=r["truth"],pred=pred,ms=elapsed/1e6))
    (folder/(device+".jsonl")).write_text("".join(json.dumps(r)+"\n" for r in rows))
    print(device,len(rows),median(r["ms"] for r in rows),flush=True)
    if all((folder/(d+".jsonl")).exists() for d in ("cuda","cpu")):
        allrows=[json.loads(line) for d in ("cuda","cpu") for line in (folder/(d+".jsonl")).read_text().splitlines()]
        (folder/"rows.jsonl").write_text("".join(json.dumps(r)+"\n" for r in allrows))
        summary={}
        for shared in (False,True):
            stats={}
            for d in ("cuda","cpu"):
                rs=[r for r in allrows if r["device"]==d and r["shared"]==shared]
                stats[d]={"n":len(rs),"correct":sum(r["truth"]==r["pred"] for r in rs),
                          "median_ms":median(r["ms"] for r in rs),"p90_ms":float(np.quantile([r["ms"] for r in rs],0.9))}
            stats["ratio_cpu_to_gpu_medians"]=stats["cpu"]["median_ms"]/stats["cuda"]["median_ms"]
            summary["shared" if shared else "private"]=stats
        (folder/"summary.json").write_text(json.dumps(summary,indent=2))
        print(summary,flush=True)
    if device=="cuda":
        (folder/"config.json").write_text(json.dumps(dict(gpu=torch.cuda.get_device_name(),torch=torch.__version__,
            cpu=cpu_name(),cpu_threads=4,fit_seed=42,sample_seed=70011,samples_per_stratum=2,
            strata=["width","operator family","edited digit count","shared/private"],warmup=4,repeat_per_edit=1,
            source="2026-11-19_方法迭代/formal_rows_42.jsonl; source_line is one-based",
            model="Qwen3-1.7B plus archived encoder LoRA; bf16 embedding on CPU and FP32 blocks on selected device",
            timing="Steady-state single edit of a previously built workspace, known target. Timed parsing, canonical digit tokenization, uncached Qwen forward batched within one literal, device transfers, CPU calibration, dirty propagation, rounded readout; explicit GPU synchronization. Excludes model loading, initial state construction, I/O. Same source rows and order both devices."),indent=2))

if __name__=="__main__": run(sys.argv[1])

