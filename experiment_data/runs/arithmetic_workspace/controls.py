"""Measure two alternative executors on the identical frozen programs (seed 42)."""
import json
import random
from collections import defaultdict
import numpy as np
import torch
from torch import nn
from iterate import Candidate, out, topology, truth, build

torch.set_num_threads(2)
torch.manual_seed(42)
rng=random.Random(42)
samples=[[] for _ in range(3)]
for i in range(300):
    w=(2,4,8)[i%3]; vals=[rng.randrange(10) for _ in range(w+1)]
    nodes=topology(w,True,rng); z=build(vals,nodes,truth)
    for k,(op,l,r) in enumerate(nodes):
        samples[op].append((z[l],z[r],z[w+1+k]))

def val_error(net,x,y):
    with torch.no_grad():return ((net(x).flatten()-y)**2).mean().item()

def trained_table(rows, op):
    # Train and select on pairs of the training corpus, using tuple-disjoint validation.
    unique=sorted(set(rows)); random.Random(67+op).shuffle(unique)
    n=max(1,len(unique)//5); val=torch.tensor(unique[:n],dtype=torch.float)
    train=torch.tensor(unique[n:],dtype=torch.float)
    scale=train[:,:2].abs().amax(0).clamp(min=1.)
    sy=train[:,2].abs().max().clamp(min=1.)
    net=nn.Sequential(nn.Linear(2,64),nn.ReLU(),nn.Linear(64,64),nn.ReLU(),nn.Linear(64,1))
    initial={k:v.detach().clone() for k,v in net.state_dict().items()}
    opt=torch.optim.AdamW(net.parameters(),lr=.002,weight_decay=0)
    x=train[:,:2]/scale; y=train[:,2]/sy; best=float("inf")
    for it in range(1500):
        ids=torch.randint(len(x),(128,))
        loss=((net(x[ids]).flatten()-y[ids])**2).mean()
        opt.zero_grad();loss.backward();opt.step()
        if (it+1)%100==0:
            err=val_error(net,val[:,:2]/scale,val[:,2]/sy)
            if err<best:
                best=err;best_steps=it+1
    net.load_state_dict(initial)
    opt=torch.optim.AdamW(net.parameters(),lr=.002,weight_decay=0)
    all_rows=torch.tensor(rows,dtype=torch.float)
    x=all_rows[:,:2]/scale;y=all_rows[:,2]/sy
    for _ in range(best_steps):
        ids=torch.randint(len(x),(128,))
        loss=((net(x[ids]).flatten()-y[ids])**2).mean()
        opt.zero_grad();loss.backward();opt.step()
    net.eval()
    return net,scale,sy

def main():
    nets=[trained_table(rows,k) for k,rows in enumerate(samples)]
    model=Candidate(42,True,True)
    groups=defaultdict(list)
    for line in (out/"formal_rows_42.jsonl").read_text().splitlines():
        r=json.loads(line);groups[(r.get("range","digits"),r["mixed"],r["width"])].append(r)
    report={}
    with torch.no_grad():
        for key,rs in groups.items():
            S=len(rs); w=key[2]; expected=np.array([r["truth"] for r in rs])
            inputs=np.array([[model.encode(n,v) for n,v in zip(model.names,r["after"])] for r in rs])
            nn_z=torch.zeros((S,3*w)); nn_z[:,:w+1]=torch.tensor(inputs)
            symbolic=np.zeros((S,3*w),dtype=np.int64);symbolic[:,:w+1]=np.array([r["after"] for r in rs])
            graphs=np.asarray([r["graph"] for r in rs]);idx=np.arange(S)
            for j in range(graphs.shape[1]):
                ops,left,right=graphs[:,j].T
                l=symbolic[idx,left];r=symbolic[idx,right]
                symbolic[:,w+1+j]=np.select([ops==0,ops==1],[l+r,l-r],default=l*r)
                for op,(net,sx,sy) in enumerate(nets):
                    sel=np.where(ops==op)[0]
                    if not len(sel):continue
                    inp=torch.stack([nn_z[sel,left[sel]],nn_z[sel,right[sel]]],1)
                    nn_z[sel,w+1+j]=net(inp/sx).flatten()*sy
            report[str(key)]=dict(n=S,mlp=int(np.sum(nn_z[:,-1].round().numpy()==expected)),
                                  exact_symbolic=int(np.sum(symbolic[:,-1]==expected)))
    (out/"controls.json").write_text(json.dumps(report,indent=2))
    print(report,flush=True)

if __name__=="__main__":main()
