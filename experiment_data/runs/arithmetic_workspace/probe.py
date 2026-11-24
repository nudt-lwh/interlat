"""Readout-only intervention on the fixed executor, using original training tuples."""
import sys,json
from pathlib import Path
import numpy as np
import torch
out=Path(__file__).resolve().parent
prev=out.parent/"neural_baselines"
sys.path.extend([str(prev),str(out.parents[2])])
from branch_data import splits,schedule,reference
from interlat.models.workspace_model import Executor

@torch.inference_mode()
def main():
    raw=np.load(out/"inputs.npz")
    feat=torch.tensor(raw["z"],device="cuda")
    names=list(raw["names"])
    ex=Executor(256,4,node_tokens=2).cuda().eval()
    state=torch.load(prev/"fork_best.pt",weights_only=True,map_location="cpu")["model"]
    ex.load_state_dict({k[9:]:v for k,v in state.items() if k.startswith("executor.")})
    train_x=[];train_y=[];dev={}
    for w in (2,4,8):
        ins,ops=schedule(w)
        train,validation,_=splits(w)
        for mode,vs in (("train",train),("dev",validation)):
            xs=[];ys=[]
            for offset in range(0,len(vs),64):
                val=vs[offset:offset+64]; b=len(val)
                idx=torch.tensor(val,device="cuda")
                z={n:feat[names.index(n),idx[:,j]][:,None,:].repeat(1,2,1) for j,n in enumerate(ins)}
                for n,_,(l,r) in ops:
                    z[n]=ex(torch.stack((z[l],z[r]),1),torch.ones(b,dtype=torch.long,device="cuda"),
                            torch.tensor([0,1],device="cuda").expand(b,2))
                xs.extend(z["g"].flatten(1).cpu().numpy())
                ys.extend(reference(w,v)["g"] for v in val)
            if mode=="train": train_x.extend(xs);train_y.extend(ys)
            else:dev[w]=(xs,ys)
    x=np.array(train_x,dtype=float)
    mu=x.mean(0); scale=x.std(0).clip(.0001); x=(x-mu)/scale
    y=np.asarray(train_y); bias=y.mean()
    metrics={}
    for alpha in (.01,1,100):
        beta=np.linalg.solve(x.T@x+alpha*np.eye(x.shape[1]),x.T@(y-bias))
        metrics[alpha]={w:float(np.mean(np.rint((np.asarray(v)-mu)/scale@beta+bias)==truth)) for w,(v,truth) in dev.items()}
    print(metrics,flush=True)
    (out/"probe.json").write_text(json.dumps(metrics,indent=2))

if __name__=="__main__":main()
