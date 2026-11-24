"""Verify returned cloud rows against the original price oracle and the saved generations."""
import argparse
import hashlib
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

import e8

HERE=Path(__file__).resolve().parent

def verify(folder):
    meta=e8.read(folder/"metadata.json")
    mode=meta["mode"]
    assert mode in e8.MODES
    assert meta["protocol_hash"]==e8.sha(HERE/"config.json")
    rows=[json.loads(line) for line in (folder/"rows.jsonl").read_text(encoding="utf8").splitlines()]
    indexed={tuple(r["key"]):r for r in rows}
    assert len(rows)==len(indexed)==192
    batches=[r for f in sorted(folder.glob("batch_*.json")) for r in e8.read(f)]
    assert len(batches)==192 and len({tuple(r["key"]) for r in batches})==192
    for b in batches:
        assert all(indexed[tuple(b["key"])][k]==v for k,v in b.items())
    summary=e8.read(folder/"summary.json")
    cases=e8.read(HERE/"cases.json")
    original=HERE.parent/"2026-11-19_P1-1/direct_rows.jsonl"
    reference={(r["pair"],r["direction"],r["step"]):r for r in map(json.loads,original.read_text(encoding="utf8").splitlines())}
    sequences=defaultdict(list)
    counts=Counter()
    for c in cases:
        simulated=list(map(int,re.findall(r"\d+",c["initial"])))
        for step,u in enumerate(c["turns"],1):
            k=c["pair"],c["direction"],step
            r=indexed[k]
            assert r["id"]==u["id"]==reference[k]["id"] and r["truth"]==reference[k]["truth"]
            assert 0<r["output_tokens"]<=meta["max_new_tokens"] and r["input_tokens"]>0
            assert not r["truncated"] or r["output_tokens"]==meta["max_new_tokens"]
            if mode=="frontend":
                edit=e8.parse_pair(r["output"])
                if edit:
                    simulated[{"guests":0,"nights":1}[edit[0]]]=edit[1]
                g,n,rate,l,s,b=simulated
                predicted=g*(n*rate+l)+n*s+b
                assert (list(edit) if edit else None)==r["edit"]
                assert bool(edit)==r["parsed"] and r["field_ok"]==bool(edit and edit[0]==u["field"])
                assert r["edit_ok"]==(edit==(u["field"],u["value"]))
                counts["field_correct"]+=r["field_ok"]
                counts["edit_correct"]+=r["edit_ok"]
            else:
                predicted=e8.amount(r["output"])
                assert r["parsed"]==(predicted is not None)
            assert predicted==r["prediction"] and r["correct"]==(predicted==reference[k]["truth"])
            counts["parsed"]+=r["parsed"]
            counts["correct_steps"]+=r["correct"]
            counts["truncated"]+=r["truncated"]
            sequences[k[:2]].append(r["correct"])
    counts.update(dict(steps=192,sequences=48,pairs=24,all_steps_correct=sum(map(all,sequences.values())),
                    final_correct=sum(v[-1] for v in sequences.values()),
                    both_finals_correct=sum(sequences[i,-1][-1] and sequences[i,1][-1] for i in range(24))))
    assert summary==dict(counts),(summary,counts)
    return dict(mode=mode,**summary,hashes={f:e8.sha(folder/f) for f in ["metadata.json","rows.jsonl","summary.json"]},
                mean_generated_tokens=sum(r["output_tokens"] for r in rows)/192,
                positive_amount_matches=sum(r["correct"] and r["parsed"] for r in rows))

if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--out",type=Path,default=HERE)
    parser.add_argument("--modes",nargs="+",choices=e8.MODES,default=list(e8.MODES))
    args=parser.parse_args()
    result={m:verify(args.out/m) for m in args.modes}
    e8.write(args.out/("verified.json" if set(args.modes)==set(e8.MODES) else "verified_"+"_".join(args.modes)+".json"),result)
    print(json.dumps(result,ensure_ascii=False,indent=2))
