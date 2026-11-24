"""Frozen confidence gate. Run at cloud project root with the archived belief experiment."""
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer

out=Path(__file__).resolve().parent
project=out.parents[2]
prior=out.parent/"neural_baselines"
sys.path[:0]=[str(prior),str(project)]
import belief_run as old
from interlat.evaluation.belief_graph import belief_revision

def load(path):
    return [json.loads(l) for l in path.read_text().splitlines()]

def dump(name,x):
    (out/name).write_text(json.dumps(x,indent=2))

def literals(rows):
    return sorted({old.EMPTY}|{t for p,c,*_ in rows for t in p+c})

def encode(texts,fixed):
    x=tok(texts,padding="max_length" if fixed else True,max_length=96,truncation=True,return_tensors="pt").to("cuda")
    h=lm(**x,use_cache=False).last_hidden_state
    m=x.attention_mask.unsqueeze(-1)
    return ((h.float()*m).sum(1)/m.sum(1)).half().cpu()

def dictionary(rows):
    texts=literals(rows)
    vectors={}
    for i in range(0,len(texts),8):
        group=texts[i:i+8]
        batch=encode(group+[old.EMPTY]*(8-len(group)),True)
        vectors.update(zip(group,batch))
    return vectors

def predict(rows,d):
    texts=sorted(d);ids={t:i for i,t in enumerate(texts)}
    features=torch.stack([d[t] for t in texts]).cuda()
    indices=torch.tensor([[ids[t] for t in p+c]+[y0,y1] for p,c,y0,y1,_ in rows],device="cuda")
    output=[]
    for batch in indices.split(32):
        a,b=old.predict(model,features,batch,ids[old.EMPTY])
        conf,p=b.softmax(-1).max(-1)
        output.extend(zip(a.argmax(-1).tolist(),p.tolist(),conf.tolist()))
    return output

def rules(row):
    p,c,*_=row
    _,final=belief_revision({"questions":"\n".join(p[:2])},{"questions":"\n".join(p)})
    order=["c" if "may or may not" in x else "b" if "does not" in x else "a" for x in c]
    return order.index(final)

def combine(a,b,prob,rule,threshold):
    return b if b!=a and prob>=threshold else rule

def metrics(rows,method):
    groups=[np.array([r[method]==r["gt1"] for r in rows if (r["gt0"]!=r["gt1"])==g]) for g in [True,False]]
    estimates=[g.mean() for g in groups]
    rng=np.random.default_rng(1977)
    draws=[rng.choice(g,(4000,len(g))).mean(1) for g in groups]
    def ci(arr):
        return np.quantile(arr,[0.025,0.975]).tolist()
    return dict(bu=float(estimates[0]),bm=float(estimates[1]),breu=float(np.mean(estimates)),
                bu_ci=ci(draws[0]),bm_ci=ci(draws[1]),breu_ci=ci((draws[0]+draws[1])/2))

def history_check(test,stable):
    main={r["id"]:r for r in load(prior/"belief_rows.jsonl")}
    both={r["id"].split("/",1)[1]:r for r in load(prior/"belief_both_rows.jsonl") if r["id"].startswith("tollens/")}
    def joint(r):return r["gt0"]!=r["gt1"] and r["initial"]==r["gt0"] and r["edited"]==r["gt1"]
    diff=[row for row in test if joint(main[row[-1]])!=joint(both[row[-1]])]
    selected={t for p,c,*_ in diff for t in p+c}|{old.EMPTY}
    records={}
    for tag,universe in [("single_modus",test),("both_modi",old.benchmark(True))]:
        d={};texts=literals(universe)
        for i in range(0,len(texts),64):
            batch=texts[i:i+64]
            if selected.intersection(batch):
                d.update({s:v for s,v in zip(batch,encode(batch,False)) if s in selected})
        records[tag]=d
    logits={k:predict(diff,d) for k,d in records.items()}
    stats={}
    for t in sorted(selected):
        mixed=encode([old.EMPTY,t]+[old.EMPTY]*6,True)[1]
        single=encode([t],True)[0]
        stats[t]={"dynamic_max_abs":(records["single_modus"][t]-records["both_modi"][t]).abs().max().item(),
                  "fixed_layout_rebatch_max_abs":(stable[t]-mixed).abs().max().item(),
                  "single_vs_fixed_max_abs":(single-stable[t]).abs().max().item()}
    return dict(original_joint=[sum(map(joint,mp.values())) for mp in (main,both)],feature_checks=stats,
                items=[dict(id=r[-1],original=[main[r[-1]],both[r[-1]]],
                    isolated_head_replay=[logits[k][i] for k in records],fixed=predict([r],{t:stable[t] for t in selected})[0]) for i,r in enumerate(diff)])

@torch.inference_mode()
def main():
    global tok,lm,model
    torch.manual_seed(42)
    tok=AutoTokenizer.from_pretrained(old.BASE)
    lm=AutoModel.from_pretrained(old.BASE,dtype=torch.bfloat16).cuda().eval()
    model=old.BeliefModel(lm.config.hidden_size).cuda().eval()
    model.load_state_dict(torch.load(prior/"belief_best.pt",weights_only=True))
    dev=old.synth(3000,True)[1::2]
    raw=predict(dev,dictionary(dev))
    candidates=[0,0.5,0.6,0.7,0.8,0.9,0.95,0.99,1.01]
    devrules=[rules(r) for r in dev]
    devscores={}
    for thr in candidates:
        groups=[[],[]]
        for (_,_,y,z,_),pred,rule in zip(dev,raw,devrules):
            groups[int(y!=z)].append(combine(*pred,rule,thr)==z)
        devscores[thr]=float(np.mean([np.mean(g) for g in groups]))
    threshold=max(candidates,key=lambda t:(devscores[t],t))
    dump("config.json",dict(dev="1500 held-out synthetic tollens examples (original dev, no Belief-R test IDs)",selection="macro BU/BM, ties choose highest threshold",grid=candidates,dev_scores=devscores,threshold=threshold,
         gate="if neural answer changes and max posterior >= threshold use edited neural answer, otherwise archived rule final",pooling="masked float32 mean; bf16 backbone, right-pad to 96 and fill batches to eight; fp16 features",bootstrap="stratified 4000, seed 1977",checkpoint_sha256=hashlib.sha256((prior/"belief_best.pt").read_bytes()).hexdigest()))
    print("locked",threshold,flush=True)
    test=old.benchmark();stable=dictionary(test);raw=predict(test,stable)
    archived={r["id"]:r for r in load(project/"experiment_data/runs/belief_graph_v4/rows.jsonl")}
    rows=[]
    for (_,_,gt0,gt1,key),(a,b,conf) in zip(test,raw):
        rule="abc".index(archived[key]["a1"])
        rows.append(dict(id=key,gt0=gt0,gt1=gt1,copy=a,neural=b,rule=rule,confidence=conf,
             used_neural=b!=a and conf>=threshold,fused=combine(a,b,conf,rule,threshold)))
    (out/"rows.jsonl").write_text("".join(json.dumps(r)+"\n" for r in rows))
    stats={k:metrics(rows,k) for k in ("copy","neural","rule","fused")}
    stats["neural_routes"]=sum(r["used_neural"] for r in rows)
    stats["n"]=len(rows)
    dump("summary.json",stats)
    dump("batch_diagnosis.json",history_check(test,stable))
    print(stats,flush=True)

if __name__=="__main__":
    main()
