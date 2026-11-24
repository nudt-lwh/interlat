from pathlib import Path
import sys
folder = Path(__file__).resolve().parent
sys.dont_write_bytecode = True
sys.path.insert(0, str(folder.parent / "edit_order"))
"""Read locked P2-2 examples and fit its identical training set; test once after selecting alpha on new development examples."""
from shared import *
import ast
from collections import Counter
from fractions import Fraction
from math import isclose

p2=folder.parent/"extended_operators"
# Import only these definitions. Importing the original script would rerun P2-2 on import.
tree=ast.parse((p2/"run.py").read_text(encoding="utf-8"))
tree.body=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in {"graph_for","exact","basis"}]
exec(compile(tree,str(p2/"run.py"),"exec"))
names=["add","sub","mul","div","gt","lt"]
masks=[[0,1,2],[0,1,2],[0,1,2,3],[0,1,2,4],[0,1,2],[0,1,2]]
model=frozen_model()
sg=random.Random(42)
groups=[[] for _ in names]
for i in range(300):
    w=(2,4,8)[i%3]
    vals=[sg.randint(1,9) for _ in range(w+1)]
    graph=graph_for(w,i%2,sg)
    gold=build(vals,graph,exact)
    for k,(op,l,r) in enumerate(graph,len(vals)):
        groups[op].append((basis(float(gold[l]),float(gold[r])),float(gold[k])))
def fitted(alpha):
    params=[]
    for op,entries in enumerate(groups):
        X,y=map(np.array,zip(*entries))
        ix=masks[op] if alpha is None else list(range(7))
        X=X[:,ix]
        if alpha is not None:
            # Augment rows to avoid squaring condition number. Intercept is unpenalized.
            penalty=np.eye(len(ix))*np.sqrt(alpha)
            penalty[0,0]=0
            X=np.vstack([X,penalty])
            y=np.concatenate([y,np.zeros(len(ix))])
        c=np.linalg.lstsq(X,y,rcond=None)[0]
        coef=np.zeros(7)
        coef[ix]=c
        params.append(coef)
    return params

def score(row,coef):
    vals=row["inputs"].copy()
    graph=row["graph"]
    target=row["target"]
    initial=build([model.encode(n,v) for n,v in zip(model.names,vals)],graph,
                  lambda op,l,r:float(np.dot(coef[op],basis(l,r))))
    vals[target]=row["new"]
    gold=build(vals,graph,exact)
    state=initial.copy()
    state[target]=model.encode(model.names[target],row["new"])
    dirty={target}
    first=None
    correct_nodes=0
    for k,(op,l,r) in enumerate(graph,len(vals)):
        if l in dirty or r in dirty:
            state[k]=float(np.dot(coef[op],basis(state[l],state[r])))
            dirty.add(k)
        ok=isclose(state[k],float(gold[k]),rel_tol=1e-9,abs_tol=1e-9) if op==3 else round(state[k])==gold[k]
        if not ok and first is None:
            first=dict(op=names[op],node=k)
        correct_nodes+=int(ok)
    assert int(gold[-1])==gold[-1]
    return dict(pred=round(state[-1]),gold=int(gold[-1]),correct_nodes=correct_nodes,nodes=len(graph),first=first)

dg=random.Random(5236951)
dev=[]
for i in range(540):
    w=(3,4,8)[i%3]
    lo,hi=((1,9),(20,39),(100,299))[(i//3)%3]
    graph=graph_for(w,(i//9)%2,dg)
    vals=[dg.randint(lo,hi) for _ in range(w+1)]
    for t in [0,dg.randrange(1,w+1)]:
        dev.append(dict(inputs=vals,graph=graph,target=t,new=change(dg,vals[t],(lo,hi))))
alphas=[1e-8,1e-6,1e-4,1e-2]
selection={}
with (folder/"train_log.jsonl").open("w") as train:
    for alpha in alphas:
        params=fitted(alpha)
        out=[score(row,params) for row in dev]
        selection[alpha]=sum(z["pred"]==z["gold"] for z in out)
        record(train,dict(alpha=alpha,dev_total=len(out),dev_correct=selection[alpha],coefficients=[p.tolist() for p in params]))
    winner=max(alphas,key=selection.get)
    masked=fitted(None)
    record(train,dict(method="literal_mask",masks=masks,coefficients=[p.tolist() for p in masked]))
artifacts(folder,train_graphs=300,train_seed=42,dev_seed=5236951,dev_edits=len(dev),dev_widths=[3,4,8],dev_ranges=[[1,9],[20,39],[100,299]],
          test_input_sha256=hashlib.sha256((p2/"rows.jsonl").read_bytes()).hexdigest(),ridge_grid=alphas,dev_correct=selection,selected_alpha=winner,
          loss="sum squared residual plus alpha times squared non-intercept coefficients; choose dev root exact then smallest alpha",masks=masks,
          node_check="all post-edit derived nodes, div 1e-9 absolute or relative tolerance, integer outputs rounded",methods=["strict masks linear gt/lt","regularized shared basis"])
methods={"strict_mask":masked,"shared_ridge":fitted(winner)}
summary={}
with (folder/"rows.jsonl").open("w") as of,(p2/"rows.jsonl").open() as inf:
    for line,r in enumerate(map(json.loads,inf),1):
        paths={}
        for label,param in methods.items():
            s=score(r,param)
            assert s["gold"]==r["truth"]
            paths[label]=s
            tall=summary.setdefault(label,dict(n=0,root_correct=0,nodes=0,correct_nodes=0,all_nodes_correct=0,first_wrong=Counter(),first_wrong_on_root_failure=Counter()))
            tall["n"]+=1
            tall["nodes"]+=s["nodes"]
            tall["correct_nodes"]+=s["correct_nodes"]
            correct=s["pred"]==s["gold"]
            tall["root_correct"]+=correct
            if s["first"] is None:tall["all_nodes_correct"]+=1
            else:
                tall["first_wrong"][s["first"]["op"]]+=1
                if not correct:tall["first_wrong_on_root_failure"][s["first"]["op"]]+=1
        record(of,dict(source_line=line,shared_ols_pred=r["pred"],symbolic=r["symbolic"],**paths))
save(folder,"summary",dict(methods=summary,selected_alpha=winner,dev_correct=selection,dev_n=len(dev),stored_baselines=json.loads((p2/"summary.json").read_text())["overall"]))
