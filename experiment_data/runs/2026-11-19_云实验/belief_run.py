"""Train a node-local neural belief workspace on synthetic scenarios."""
import collections
import glob
import json
import random
from pathlib import Path

import torch
from torch import nn
from transformers import AutoModel, AutoTokenizer
from interlat.models.workspace_model import Executor

OUT = Path(__file__).resolve().parent
BASE = "experiment_data/artifacts/Qwen3-1.7B"
EMPTY = "[no additional premise]"
# action, outcome, enabling condition, alternative action; held-out settings last.
SCENES = [
 ("scan the ticket","enter the concert","the scanner is working","show a guest pass"),
 ("turn the ignition key","start the engine","the battery is charged","use the remote starter"),
 ("heat the kettle","boil the water","the stove is working","use an immersion heater"),
 ("send a wire transfer","pay the bill","the account has sufficient funds","pay in cash"),
 ("press the elevator button","reach the upper floor","the lift has power","take the stairs"),
 ("dial the office number","reach the clerk","the phone line is connected","visit the front desk"),
 ("insert the key","open the door","the lock is functional","use an access card"),
 ("submit the form online","register for the course","the website is accepting requests","register in person"),
 ("print the report","obtain a paper copy","the printer has toner","use a photocopier"),
 ("buy a train ticket","travel to the city","the train service is running","take a coach"),
 ("plant the seeds","grow the herbs","the soil is moist","transplant young seedlings"),
 ("plug in the charger","charge the phone","the socket has power","use a wireless charging pad"),
 ("upload the photos","deliver the pictures","the network is connected","send a memory card"),
 ("ring the bell","alert the occupant","the bell is connected","knock on the window"),
 ("swipe the voucher","receive the discount","the voucher is valid","show a membership card"),
 ("switch on the projector","display the slides","the projector has a lamp","use a large monitor"),
 ("pull the handle","release the latch","the linkage is intact","use the emergency lever"),
 ("open the tap","fill the bucket","the water supply is on","collect rainwater"),
 ("book a hotel room","secure lodging","the hotel has vacancies","stay at a hostel"),
 ("light a match","ignite the fire","the match is dry","use a lighter"),
]


def synth(n, dev=False):
    rng=random.Random(194 if dev else 193)
    scenes=SCENES[-4:] if dev else SCENES[:-4]
    rows=[]
    for i in range(n):
        action,outcome,condition,alternative=rng.choice(scenes)
        who=rng.choice(("Mira","Ravi","Noor","Tao","Lena","Ivo","Ada","Niko"))
        where=rng.choice(("at the station","near the workshop","during the visit",
                          "on Monday","in the morning","during the demonstration","after lunch","at noon"))
        p=f"{who} decides to {action} {where}"
        q=f"{who} manages to {outcome}"
        mt=bool(i%2)
        update=bool(i//2%2)
        second=f"{who} does not manage to {outcome}" if mt else p
        verb=("decide to "+action+" "+where) if mt else ("manage to "+outcome)
        cands=[who+" "+verb.replace("decide ","decides ",1).replace("manage ","manages ",1)+".",
               who+" does not "+verb+".",who+" may or may not "+verb+"."]
        if update:
            extra=rng.choice([condition, f"{who} checks that {condition}", f"{who} ensures that {condition}"])
        else:
            extra=rng.choice([f"{who} decides to {alternative} {where}",
                              f"{who} chooses to {alternative}", f"there is another opportunity to {alternative}"])
        texts=[f"If {p}, then {q}",second,f"If {extra}, then {q}"]
        order=list(range(3)); rng.shuffle(order)
        rows.append((texts,[cands[j] for j in order],order.index(int(mt)),
                     order.index(2 if update else int(mt)),str(i)))
    return rows


def benchmark(include_both=False):
    from datasets import Dataset
    source=Dataset.from_file(next((Path.home()/".cache/huggingface/datasets/CAiRE___belief_r").rglob("*test.arrow")))
    pairs=collections.defaultdict(dict)
    for e in source: pairs[(e["dataset_id"],e["modus"])][e["step"]]=e
    archive="experiment_data/runs/belief_graph_v4/rows.jsonl"
    rows=[]
    archived=[json.loads(line) for line in open(archive)]
    wanted=[(r["id"],"tollens",r) for r in archived]
    if include_both:
        wanted=[(i,m,None) for (i,m),steps in pairs.items() if len(steps)==2]
    for key,modus,old in wanted:
        a,b=(pairs[key,modus][t] for t in ("time_t","time_t1"))
        def premises(e):
            return [v.strip() for v in e["questions"].splitlines() if v.strip() and not v.strip().startswith(("What","("))]
        x,y=premises(a)[:2],premises(b)[:3]
        assert len(y)==3 and x==y[:2],key
        if old: assert old["gt0"]==a["ground_truth"] and old["gt1"]==b["ground_truth"],key
        assert all(a[k]==b[k] for k in "abc"),key
        rows.append((y,[a[k] for k in "abc"],"abc".index(a["ground_truth"]),"abc".index(b["ground_truth"]),f"{modus}/{key}" if include_both else key))
    assert len(rows)==(1282 if include_both else 641)
    return rows


class BeliefModel(nn.Module):
    def __init__(self, hidden):
        super().__init__()
        self.encode=nn.Sequential(nn.LayerNorm(hidden),nn.Linear(hidden,512))
        self.execute=Executor(dim=256,layers=4,node_tokens=2)
        self.belief_op=nn.Parameter(torch.zeros(1,1,256))
        self.read=nn.Sequential(nn.Linear(512,256),nn.GELU(),nn.Linear(256,1))

    def read_state(self,z,c):
        roles=torch.arange(3,device=z.device).expand(len(z),3)
        parents=self.execute.embed_parents(z,roles)+self.belief_op
        out=self.execute.forward_single(parents,roles[:,0]).mean(1)[:,None,:]
        v=c.mean(-2)
        return self.read(torch.cat([v*out,torch.abs(v-out)],dim=-1)).squeeze(-1)

    def states(self,x):
        return self.encode(x).unflatten(-1,(2,256))


def embeddings(texts):
    tok=AutoTokenizer.from_pretrained(BASE)
    lm=AutoModel.from_pretrained(BASE,dtype=torch.bfloat16).cuda().eval()
    result=[]
    with torch.inference_mode():
        for i in range(0,len(texts),64):
            x=tok(texts[i:i+64],padding=True,truncation=True,max_length=96,return_tensors="pt").to("cuda")
            h=lm(**x,use_cache=False).last_hidden_state
            mask=x.attention_mask.unsqueeze(-1)
            result.append(((h.float()*mask).sum(1)/mask.sum(1)).cpu().half())
    del lm
    torch.cuda.empty_cache()
    return torch.cat(result)


def prepare(rows):
    texts=sorted({EMPTY}|{t for p,c,*_ in rows for t in p+c})
    ids={s:i for i,s in enumerate(texts)}
    features=embeddings(texts).cuda()
    index=torch.tensor([[ids[t] for t in p+c]+[y0,y1] for p,c,y0,y1,_ in rows],device="cuda")
    return index,features,ids[EMPTY]


def predict(model,features,batch,empty,rebuild=False):
    ix=batch[:,:3]
    initial=ix.clone()
    initial[:,2]=empty
    z=model.states(features[initial].float())
    c=model.states(features[batch[:,3:6]].float())
    a=model.read_state(z,c)
    fresh=model.states(features[ix[:,2:3]].float())
    edited=torch.cat((z[:,:2],fresh),1)
    b=model.read_state(edited,c)
    if rebuild:
        return a,b,model.read_state(model.states(features[ix].float()),c)
    return a,b


def main():
    torch.manual_seed(42)
    train,dev=synth(30000),synth(3000,True)
    index,features,empty=prepare(train+dev)
    model=BeliefModel(features.shape[1]).cuda()
    opt=torch.optim.AdamW(model.parameters(),lr=1e-4)
    best=-1
    for epoch in range(3):
        model.train()
        for ids in torch.randperm(len(train),device="cuda").split(64):
            batch=index[ids]
            a,b=predict(model,features,batch,empty)
            loss=(nn.functional.cross_entropy(a,batch[:,6])+nn.functional.cross_entropy(b,batch[:,7]))/2
            opt.zero_grad();loss.backward();nn.utils.clip_grad_norm_(model.parameters(),1);opt.step()
        model.eval()
        with torch.no_grad():
            joint=[]; change=[]; initial=[]
            for batch in index[len(train):].split(64):
                a,b=predict(model,features,batch,empty)
                ok=a.argmax(1)==batch[:,6]
                initial.extend(ok.tolist())
                joint.extend((ok & (b.argmax(1)==batch[:,7])).tolist())
                change.extend((batch[:,6]!=batch[:,7]).tolist())
            groups=[sum(j for j,d in zip(joint,change) if d==v)/change.count(v) for v in (True,False)]
            print("dev",epoch,sum(initial)/len(initial),groups,flush=True)
            if sum(groups)>best:
                best=sum(groups);torch.save(model.state_dict(),OUT/"belief_best.pt")
    model.load_state_dict(torch.load(OUT/"belief_best.pt",weights_only=True))
    score_external(model,benchmark(),"belief_rows.jsonl")
    score_external(model,benchmark(True),"belief_both_rows.jsonl")


def score_external(model,rows,filename):
    index,features,empty=prepare(rows)
    with torch.no_grad(),(OUT/filename).open("w") as f:
        for start in range(0,len(rows),64):
            batch=index[start:start+64]
            preds=[v.argmax(1).tolist() for v in predict(model,features,batch,empty,True)]
            for k,(_,_,y0,y1,key) in enumerate(rows[start:start+64]):
                f.write(json.dumps(dict(id=key,gt0=y0,gt1=y1,initial=preds[0][k],
                                        edited=preds[1][k],rebuild=preds[2][k]))+"\n")
    print("done",len(rows),flush=True)


if __name__=="__main__": main()
