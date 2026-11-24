"""Independently reprice examples and test rows; check denominators and sample identity."""
import hashlib
import json
import re
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
P1 = HERE.parent/"hotel_replay"

def load(path):
    return json.loads(path.read_text(encoding="utf8"))

def total(state):
    g,n,r,l,s,b=state
    return g*(n*r+l)+n*s+b

def checked_states(case):
    state=list(map(int,re.findall(r"\d+",case["initial"])))
    for u in case["turns"]:
        state[{"guests":0,"nights":1}[u["field"]]]=u["value"]
        yield total(state)

def main():
    for f,d in load(HERE/"config.json")["locked_sha256"].items():
        assert hashlib.sha256((HERE/f).read_bytes()).hexdigest()==d
    examples=load(HERE/"demos.json")
    for d in examples:
        assert list(checked_states(d))[-1]==int(d["messages"][-1]["content"])==d["gold"]
    cases=load(P1/"cases.json")
    did={u["id"].split("/")[0] for c in examples for u in c["turns"]}
    tid={u["id"].split("/")[0] for c in cases for u in c["turns"]}
    assert len(examples)==4 and len(did)==6 and not tid.intersection(did)
    q_rows=[json.loads(l) for l in (HERE/"rows.jsonl").read_text(encoding="utf8").splitlines()]
    zero=[json.loads(l) for l in (P1/"direct_rows.jsonl").read_text(encoding="utf8").splitlines()]
    assert len(q_rows)==len(zero)==192
    assert all((a["pair"],a["direction"],a["step"],a["truth"],a["id"])==
               (b["pair"],b["direction"],b["step"],b["truth"],b["id"]) for a,b in zip(q_rows,zero))
    index={(r["pair"],r["direction"],r["step"]):r for r in q_rows}
    assert len(index)==192
    seq=defaultdict(list)
    for c in cases:
        for step,gold in enumerate(checked_states(c),1):
            r=index[c["pair"],c["direction"],step]
            assert r["truth"]==gold and r["correct"]==(r["prediction"]==gold) and r["ns"]>0
            seq[c["pair"],c["direction"]].append(r["correct"])
    summary=load(HERE/"summary.json")
    assert summary["correct_steps"]==sum(sum(x) for x in seq.values())
    assert summary["all_steps_correct"]==sum(all(x) for x in seq.values())
    assert summary["final_correct"]==sum(x[-1] for x in seq.values())
    assert summary["both_finals_correct"]==sum(seq[i,-1][-1] and seq[i,1][-1] for i in range(24))
    assert summary["numeric"]==sum(r["prediction"] is not None for r in q_rows)
    proof=dict(steps_checked=len(q_rows),source_alignment_with_zero_shot=True,
        gold_checksum_verified=True,development_bills_checked=len(examples),
        duplicate_demo_test_customer_sentences=sorted(
        {u["text"] for d in examples for u in d["turns"]}&{u["text"] for d in cases for u in d["turns"]}))
    (HERE/"verified.json").write_text(json.dumps(proof,indent=2),encoding="utf8")
    print(json.dumps(summary,indent=2))

if __name__=="__main__":
    main()
