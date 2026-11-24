from pathlib import Path
import sys
folder = Path(__file__).resolve().parent
sys.dont_write_bytecode = True
sys.path.insert(0, str(folder.parent / "2026-11-19_P0-2"))
from shared import *
from fractions import Fraction
from statistics import mean

# The first two branches implement s/x and y/s. Their first parent compares the
# ratios, giving integer roots without rounding ground truth or rejecting test cases.
# Other branches include a direct comparison plus ordinary arithmetic.
def graph_for(w, reverse, rng):
    leaves = list(range(1,w+1))
    rng.shuffle(leaves)
    graph = [(3,0,leaves[0]),(3,leaves[1],0)]
    for i in range(2,w):
        op = (4 if reverse else 5) if i==2 else rng.randrange(3)
        graph.append((op,0,leaves[i]))
    first = w+1
    graph.append((5 if reverse else 4,first,first+1))
    level = [first+len(graph)-1] + list(range(first+2,first+w))
    rng.shuffle(level)
    while len(level)>1:
        new = []
        for i in range(0,len(level)-1,2):
            new.append(first+len(graph))
            graph.append((rng.randrange(2),level[i],level[i+1]))
        if len(level)%2:
            new.append(level[-1])
        level = new
    return graph

def exact(op,l,r):
    if op == 3:
        return Fraction(l,r)
    if op == 4:
        return int(l>r)
    if op == 5:
        return int(l<r)
    return truth(op,l,r)

def basis(l,r):
    return [1.,l,r,l*r,l/r if r != 0 else 0.,float(l>r),float(l<r)]

def fit_extended():
    rng = random.Random(42)
    samples = [[] for _ in range(6)]
    for i in range(300):
        w = (2,4,8)[i%3]
        vals = [rng.randint(1,9) for _ in range(w+1)]
        graph = graph_for(w,i%2,rng)
        z = build(vals,graph,exact)
        for k,(op,l,r) in enumerate(graph,w+1):
            samples[op].append((basis(float(z[l]),float(z[r])),float(z[k])))
    coefs = []
    with (folder/"train_log.jsonl").open("w") as f:
        for op,pairs in enumerate(samples):
            X,Y = map(np.array,zip(*pairs))
            coef,_,rank,_ = np.linalg.lstsq(X,Y,rcond=None)
            record(f,dict(op=op,n=len(pairs),rank=int(rank),coefficients=coef.tolist(),train_max_abs=float(np.max(np.abs(X@coef-Y)))))
            coefs.append(coef)
    return coefs

artifacts(folder, fit="300 distinct generated training graphs on widths 2,4,8 and positive single digits; frozen input/decimal decoder; per-operator least squares without post-fit thresholding",
         design="two quotient branches s/x, y/s feed gt or lt; third branch uses complementary direct comparison, other branches and reductions use arithmetic. No undefined division is tested. Rational oracle; all roots integers by construction. Positive inputs; two independent edits from the same initial state.",
         basis=["1","l","r","l*r","l/r if r != 0 else 0","gt","lt"],
         division_diagnostic="np.isclose with rtol=1e-9, atol=1e-9",data_seed=22719)
model = frozen_model()
coefficients = fit_extended()
model.execute = lambda op,l,r: float(np.dot(coefficients[op],basis(l,r)))
counts = defaultdict(int)
cells = defaultdict(lambda:defaultdict(int))
by_op = defaultdict(lambda:defaultdict(int))
with (folder/"rows.jsonl").open("w") as f:
    for name,(lo,hi) in ranges.items():
        low = max(1,lo)
        for w in (5,7,12):
            for reverse in (False,True):
                rng = random.Random(22719 + 100*w + 10000*int(reverse) + low)
                for i in range(300):
                    graph = graph_for(w,reverse,rng)
                    vals = [rng.randint(low,hi) for _ in range(w+1)]
                    initial = rebuild(model,vals,graph)
                    for tgt in (0,rng.randrange(1,w+1)):
                        new = change(rng,vals[tgt],(low,hi))
                        local,after,trace = edit(model,initial,vals,graph,tgt,new,oracle=exact)
                        oracle = build(after,graph,exact)
                        # Separately implement the symbolic baseline through incremental propagation.
                        symbolic = build(vals,graph,exact)
                        symbolic[tgt] = new
                        changed = {tgt}
                        for k,(op,l,r) in enumerate(graph,len(vals)):
                            if l in changed or r in changed:
                                symbolic[k] = exact(op,symbolic[l],symbolic[r])
                                changed.add(k)
                        diag = []
                        for k,(op,l,r) in enumerate(graph,len(vals)):
                            if op < 3:
                                continue
                            match = bool(np.isclose(local[k],float(oracle[k]),rtol=1e-9,atol=1e-9)) if op==3 else round(local[k])==oracle[k]
                            tie = bool(oracle[l]==oracle[r]) if op in (4,5) else None
                            by_op[op]["n"]+=1
                            by_op[op]["correct"]+=match
                            if tie:
                                by_op[op]["ties"]+=1
                                by_op[op]["ties_correct"]+=match
                            diag.append(dict(node=k,op=op,truth=str(oracle[k]),value=local[k],match=match,tie=tie))
                        symbol_ok = int(symbolic[-1]==oracle[-1])
                        record(f,dict(range=name,width=w,reverse=reverse,task=i,inputs=vals,graph=graph,**trace,
                                      symbolic=int(symbolic[-1]),node_diagnostics=diag))
                        for stat in (counts,cells[str(w)],):
                            stat["n"]+=1
                            stat["correct"]+=trace["correct"]
                            stat["agreement"]+=trace["agreement"]
                            stat["preserved"]+=trace["preserved"]
                            stat["symbolic_correct"]+=symbol_ok
save(folder,"summary",dict(overall=counts,by_width=cells,by_new_operator=by_op))