import argparse
import json
import os
import random
import time
from pathlib import Path
from types import SimpleNamespace

import torch
from branch_data import changed, reference, schedule, splits
from interlat.training import collate as C
from interlat.training.train import save_trainable
from interlat.models.workspace_model import value_to_labels
from interlat.models.neural_edit import NeuralWorkspace, neural_edit, read_query
from interlat.evaluation.eval_edit import load_model

OUT = Path(__file__).resolve().parent
WIDTHS = (2,4,8)
for w in WIDTHS:
    names, ops = schedule(w)
    C.FAMILIES[f"fork{w}"] = dict(inputs=names, derived=ops, queries={"result":("g",)})


def record(w, v):
    fam = f"fork{w}"
    states = reference(w,v)
    labels = []
    for x in states.values():
        sign, digits = value_to_labels(x)
        labels.append([sign, *digits])
    return dict(family=fam,input_texts=[f"{n} = {x}" for n,x in zip(schedule(w)[0],v)],
                state_labels=labels,query_text="result?",answer_text=str(states["g"]),
                query_out_idx=[len(states)-1],query_out_names=["g"])


def get_model(ckpt):
    args = SimpleNamespace(model="experiment_data/artifacts/Qwen3-1.7B",ckpt=ckpt,
                           lora_r=16,node_dim=256,node_tokens=2,exec_layers=4)
    return load_model(args,"cuda")[0]


@torch.no_grad()
def score(model, dev):
    model.eval()
    result = {}
    for w, values in dev.items():
        ok = 0
        for i in range(0,len(values),16):
            g = C.collate([record(w,v) for v in values[i:i+16]],"cuda")
            z = model.execute_all(model.encode_inputs(g.input_texts,"cuda"),g)
            ids = model.read_answers(z,g,"cuda",max_new_tokens=8,generate=True)
            texts = model.tok.batch_decode(ids,skip_special_tokens=True)
            ok += sum(t.strip()==a for t,a in zip(texts,g.answer_texts))
        result[w] = ok/len(values)
    return result


def train():
    rng = random.Random(42)
    torch.manual_seed(42)
    model = get_model("experiment_data/runs/recur_s3_chain/ckpt_step900.pt").float()
    ds = {w:splits(w) for w in WIDTHS}
    dev = {w:ds[w][1] for w in WIDTHS}
    print("transfer_dev", score(model,dev), flush=True)
    params = [p for n,p in model.named_parameters() if "lora_" in n or not n.startswith("backbone.")]
    opt = torch.optim.AdamW(params,lr=5e-5)
    micro = int(os.environ.get("FORK_BATCH","8"))
    accum = 64//micro
    best = -1
    for epoch in range(3):
        batches = []
        for w in WIDTHS:
            items = list(ds[w][0])
            rng.shuffle(items)
            batches += [(w,items[i:i+micro]) for i in range(0,len(items),micro)]
        rng.shuffle(batches)
        model.train()
        for k,(w,items) in enumerate(batches,1):
            g = C.collate([record(w,v) for v in items],"cuda")
            with torch.autocast("cuda",dtype=torch.bfloat16):
                loss,_ = model(g,"cuda",lambda_s=2)
            if not torch.isfinite(loss):
                raise ValueError("nonfinite loss")
            (loss/accum).backward()
            if k%accum==0 or k==len(batches):
                torch.nn.utils.clip_grad_norm_(params,1)
                opt.step()
                opt.zero_grad()
            if k%100==0: print("train",epoch,k,round(loss.item(),3),flush=True)
        acc = score(model,dev)
        print("epoch_dev",epoch,acc,flush=True)
        if sum(acc.values())>best:
            best=sum(acc.values())
            save_trainable(model,OUT/"fork_best.pt",epoch+1)


@torch.inference_mode()
def evaluate():
    model=get_model(OUT/"fork_best.pt").float()
    rng=random.Random(745)
    with (OUT/"fork_rows.jsonl").open("w") as f:
        for w in WIDTHS:
            lay=C.family_layout(f"fork{w}")
            names=lay["names"]
            warm=C.collate([record(w,v) for v in splits(w)[1][:1]],"cuda")
            for _ in range(3):
                sample=NeuralWorkspace.build(model,warm,"cuda",lay)
                neural_edit(model,sample,"s",2,"cuda")
                read_query(model,sample,"result?",["g"],"cuda")
            torch.cuda.synchronize()
            for idx,v in enumerate(splits(w)[2]):
                g=C.collate([record(w,v)],"cuda")
                for pos in (rng.randrange(1,w+1),0):
                    vv=changed(v,pos,rng)
                    gg=C.collate([record(w,vv)],"cuda")
                    ws=NeuralWorkspace.build(model,g,"cuda",lay)
                    before=ws.z.clone()
                    routes=["local","rebuild"]
                    rng.shuffle(routes)
                    row=dict(width=w,task=idx,private=bool(pos),values=v,after=vv,truth=gg.answer_texts[0])
                    for route in routes:
                        torch.cuda.synchronize()
                        start=time.perf_counter()
                        if route=="local":
                            stat=neural_edit(model,ws,names[pos],vv[pos],"cuda")
                            target=ws
                        else:
                            target=NeuralWorkspace.build(model,gg,"cuda",lay)
                            rebuilt=target
                        torch.cuda.synchronize()
                        mid=time.perf_counter()
                        row[route]=read_query(model,target,"result?",["g"],"cuda").strip()
                        torch.cuda.synchronize()
                        row[route+"_ms"]=(time.perf_counter()-start)*1000
                        row[route+"_update_ms"]=(mid-start)*1000
                    expected = {pos}
                    for j,(_,pa,_) in enumerate(lay["derived"],start=w+1):
                        if expected.intersection(pa): expected.add(j)
                    row["expected"]=[names[j] for j in sorted(expected-{pos})]
                    row["executed"]=[names[j] for j in range(w+1,len(names)) if ws.versions[j]>0]
                    row["preserved"]=all(torch.equal(before[0,j],ws.z[0,j]) and ws.versions[j]==0
                                         for j in range(len(names)) if j not in expected)
                    if row["local"]!=row["rebuild"]:
                        diffs=(ws.z-rebuilt.z).abs().amax(dim=(-2,-1))[0].tolist()
                        row["max_abs_diff_by_node"]=dict(zip(names,diffs))
                    f.write(json.dumps(row)+"\n")
                if idx%25==0: f.flush();print("eval",w,idx,flush=True)


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("phase",choices=["train","test"])
    if parser.parse_args().phase=="train": train()
    else: evaluate()
