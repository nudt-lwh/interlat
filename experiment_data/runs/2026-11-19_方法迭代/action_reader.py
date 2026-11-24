"""Learn a small action reader from calibration utterances; execute the selected policy."""
import hashlib,json,re,sys
import numpy as np
import torch
from transformers import AutoModel,AutoTokenizer
from procurement import cases
from iterate import Candidate,out

# Training utterances are independent of generated notices. Each label describes an executable action.
def exemplars():
    data=[]
    for i,(me,other) in enumerate([("Vendor Alpha","Vendor Beta"),("Vendor Beta","Vendor Alpha"),("first","second"),("second","first")]):
        if i<2:
            phrases=[f"Authorize {me}'s offer.",f"Proceed with {me}.",f"Choose {me} for the purchase.",
                     f"{me} wins the contract.",f"{other} has lost the bid.",f"Reject {other}. Choose its rival.",
                     f"The offer from {other} is cancelled.",f"{other}'s suggestion was denied. Give its rival the deal."]
        else:
            phrases=[f"Authorize the {me} proposal.",f"The {me} entry wins.",f"Proceed with the {me} bid.",
                     f"The {other} option is rejected.",f"Discard the {other} bid. Process its rival.",
                     f"Ignore the {other} quotation.",f"We will take the {me} item in the list."]
        data.extend((s,i) for s in phrases)
    for label,phrases in [(4,["Prioritize the fastest delivery.","Use the earlier shipment.","We require the shortest lead time.",
                              "Choose fewer shipping days.","The quickest supplier gets the order.","Take the bid arriving first."]),
                          (5,["Prefer the slowest delivery.","Use the later shipment.","We require the longest lead time.",
                              "Choose more shipping days.","The slowest supplier gets the order.","Take the bid arriving last."])]:
        data.extend((s,label) for s in phrases)
    return data

def normalized(r):
    s=r["decision"]
    for p,name in zip(r["proposals"],["Vendor Alpha","Vendor Beta"]):
        s=s.replace(p[0],name)
    return s

def main():
    phase=sys.argv[1]
    if phase=="formal":
        lock=json.loads((out/"procurement_frozen.json").read_text())
        assert lock["action_sha"]==hashlib.sha256((out/"action_reader.py").read_bytes()).hexdigest()
    rows=cases(phase);train=exemplars()
    texts=sorted(set([s for s,_ in train]+[normalized(r) for r in rows]))
    base=out.parents[2]/"experiment_data/artifacts/Qwen3-1.7B"
    tok=AutoTokenizer.from_pretrained(base)
    model=AutoModel.from_pretrained(base,dtype=torch.bfloat16).cuda().eval()
    features={}
    with torch.inference_mode():
        for i in range(0,len(texts),16):
            batch=texts[i:i+16]
            t=tok([s+" The intended purchasing instruction is" for s in batch],padding=True,return_tensors="pt").to("cuda")
            h=model(**t,use_cache=False).last_hidden_state.float()
            # Position is independent of left/right padding.
            pos=(torch.arange(h.shape[1],device="cuda")[None]*t.attention_mask).amax(1)
            last=h[torch.arange(len(batch)),pos].cpu().double().numpy()
            features.update(zip(batch,last))
    x=np.stack([features[s] for s,_ in train]);mu=x.mean(0)
    x=x-mu
    scale=np.linalg.norm(x,axis=1).mean()
    x=x/scale
    y=np.eye(6)[[c for _,c in train]]
    weights=x.T@np.linalg.solve(x@x.T+.01*np.eye(len(x)),y-y.mean(0))
    typed=Candidate(42,True,True)
    for r in rows:
        p=int(np.argmax((features[normalized(r)]-mu)/scale@weights+y.mean(0)))
        fast=int(r["days"][1]<r["days"][0])
        idx=[0,1,r["first_offer"],1-r["first_offer"],fast,1-fast][p]
        r["policy"]=p;r["pred"]=idx
        _,field,value,_=r["proposals"][idx]
        qty,price,fee=[typed.encode("s",r[k]) for k in ("qty","price","fee")]
        old=typed.execute(2,qty,price)
        if field=="unit price":old=typed.execute(2,qty,typed.encode("s",value))
        else:fee=typed.encode("s",value)
        r["pred_total"]=round(typed.execute(0,old,fee))
    results={}
    for g in range(3):
        rs=[r for r in rows if r["group"]==g]
        results[g]=dict(n=len(rs),correct_total=sum(r["pred_total"]==r["total"] for r in rs),
                        both=sum(all(r["pred_total"]==r["total"] for r in rs if r["pair"]==i) for i in set(r["pair"] for r in rs)))
    print(results,flush=True)
    (out/f"action_{phase}.jsonl").write_text("".join(json.dumps(r)+"\n" for r in rows))
    (out/f"action_{phase}_summary.json").write_text(json.dumps(results,indent=2))

if __name__=="__main__":main()
