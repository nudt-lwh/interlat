"""Evaluate arithmetic answers and pair identical literals across two batch compositions."""
from pathlib import Path
import sys
from collections import defaultdict
from functools import lru_cache
sys.path.insert(0,str(Path(__file__).resolve().parent.parent/"2026-11-20_E9_共享编辑误差"))
from evaluate import calibration
from extract import *

def main():
    z=np.load(E7/"features.npz")
    names=list(z["texts"])
    assert names[:10]==[f"s = {i}" for i in range(10)]
    model=Candidate(42,True,True)
    rows=formal_rows()
    summary={};delta={}
    feature_rows=[]
    for method_ in ("dynamic","fixed"):
        a=z[method_+"_archive"]; b=z[method_+"_remixed"]
        diff=np.max(np.abs(a-b),axis=1)
        delta[method_]=dict(n=len(diff),identical=int(sum(diff==0)),maximum=float(max(diff)),
                            canonical_identical=int(sum(diff[:10]==0)),canonical_maximum=float(max(diff[:10])))
        feature_rows.extend(dict(regime=method_,text=n,max_abs=float(d)) for n,d in zip(names,diff))
    (E7/"feature_rows.jsonl").write_text("".join(json.dumps(r)+"\n" for r in feature_rows))
    with (E7/"rows.jsonl").open("w") as f:
      for tag in ("dynamic_archive","dynamic_remixed","fixed_archive","fixed_remixed"):
        cals={"archived":(model.mean,model.weights,model.bias)}
        if tag.startswith("fixed"):cals["matched"]=calibration(z["fixed_archive"][:10])
        for label,(mean,w,bias) in cals.items():
            digits=(z[tag][:10]-mean)@w+bias
            @lru_cache(None)
            def encode(value):
                return number(model,[digits[int(d)] for d in str(value)])
            cells=defaultdict(lambda:dict(n=0,correct=0,agree=0,preserved=0))
            for idx,r in enumerate(rows,1):
                vals=[encode(v) for v in r["inputs"]]; graph=r["graph"]; target=r["target"]
                state=build(vals,graph,model.execute)
                new=encode(r["after"][target]); local,dirty=edit(model,state,target,new,graph,len(vals))
                vals[target]=new
                pred=round(local[-1]); rebuilt=round(build(vals,graph,model.execute)[-1])
                preserved=all(v==local[j] for j,v in enumerate(state) if j not in dirty)
                key=f"{r['width']}/"+("mixed" if r["mixed"] else "add")+"/"+("shared" if target==0 else "private")+"/"+r.get("range","single")
                for group in ("all",key):
                    cell=cells[group];cell["n"]+=1;cell["correct"]+=int(pred==r["truth"]);cell["agree"]+=int(pred==rebuilt);cell["preserved"]+=int(preserved)
                f.write(json.dumps(dict(source_line=idx,group=key,condition=tag,calibration=label,truth=r["truth"],pred=pred,rebuild=rebuilt,preserved=preserved))+"\n")
            summary[tag+"/"+label]=dict(cells)
    (E7/"summary.json").write_text(json.dumps(dict(feature_differences=delta,accuracy=summary),indent=2))
    (E7/"config.json").write_text(json.dumps(dict(seed=42,device="RTX 4060 Laptop GPU, FP32 encoder blocks as P0-1",source="Full formal_rows_42.jsonl; graph, edits and reference as archived",
        extraction="Adjacent E9 extract.py cuda writes features.npz for all 170 literals. Archive order is s then x0..15, digits 0–9. Remixed uses fixed shuffle seed 194720 and two distractors, including one longer string. Dynamic native batch size is 12 vs 8 and sequence length varies. Fixed pads both orders and final tails to batch 8 x length 96 with masks.",
        scoring="All 170 projected features compared directly. Formal pipeline uses only s = d canonical digit vectors. Matched readout fit on ten labeled canonical digits from fixed archive composition only. Decimal recurrence and operators unchanged. Rows contain one record per original edit per condition.",training_test="No formal graph label used in fitting."),indent=2))
    print(delta);print([(k,v["all"]) for k,v in summary.items()])
if __name__=="__main__":main()
