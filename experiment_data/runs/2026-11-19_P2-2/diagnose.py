from pathlib import Path
import sys
folder = Path(__file__).resolve().parent
sys.dont_write_bytecode = True
sys.path.insert(0, str(folder.parent / "2026-11-19_P0-2"))
"""Diagnose the first numerically wrong integer node on every failed extension example."""
from collections import Counter
from fractions import Fraction
from shared import *
model=frozen_model()
with (folder/"train_log.jsonl").open() as f:
    coefs=[np.array(json.loads(line)["coefficients"]) for line in f]
def features(l,r):
    return [1.,l,r,l*r,l/r if r!=0 else 0.,float(l>r),float(l<r)]
def oracle(op,l,r):
    if op==3: return Fraction(l,r)
    if op==4: return int(l>r)
    if op==5: return int(l<r)
    return truth(op,l,r)
totals=Counter()
range_n=Counter()
range_err=Counter()
first_op=Counter()
first_cause=Counter()
examples={}
with (folder/"rows.jsonl").open() as f:
    for line in f:
        row=json.loads(line)
        range_n[row["range"]]+=1
        vals=row["inputs"].copy()
        vals[row["target"]]=row["new"]
        gold=list(vals)
        state=[model.encode(n,v) for n,v in zip(model.names,vals)]
        first=None
        any_cmp_err=False
        for j,(op,l,r) in enumerate(row["graph"],len(vals)):
            a,b=state[l],state[r]
            phi=features(a,b)
            reference=oracle(op,gold[l],gold[r])
            y=float(np.dot(coefs[op],phi))
            if op in (4,5) and round(y)!=reference:
                any_cmp_err=True
            if first is None and op!=3 and round(y)!=reference:
                first=dict(op=op,node=j,reference=str(reference),prediction=y,right_symbolic=str(gold[r]),
                           right_float=b,left_float=a,terms=list(map(float,np.array(phi)*coefs[op])))
            state.append(y)
            gold.append(reference)
        assert row["pred"]==round(state[-1]) and row["truth"]==gold[-1]
        if row["pred"]==row["truth"]:
            continue
        totals["failed"]+=1
        range_err[row["range"]]+=1
        totals["failed_with_any_comparison_error"]+=any_cmp_err
        first_op[first["op"]]+=1
        if first["op"] in (0,1,2) and first["right_symbolic"]=="0" and abs(first["right_float"]) < 1e-8 and abs(first["terms"][4])>=0.5:
            cause="spurious_quotient_term_on_near_zero_right_operand"
        elif first["op"] in (4,5):
            cause="comparison"
        else:
            cause="other"
        first_cause[cause]+=1
        if cause not in examples:
            examples[cause]=dict(**{k:row[k] for k in ("range","width","reverse","task","target","truth","pred")},first=first)
save(folder,"diagnosis",dict(total=totals,by_range={k:dict(n=v,wrong=range_err[k]) for k,v in range_n.items()},
                             first_wrong_integer_operator=first_op,first_wrong_cause=first_cause,examples=examples))
