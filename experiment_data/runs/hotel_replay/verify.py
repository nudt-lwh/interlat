"""Audit results from source IDs and an independent price calculator."""
import hashlib
import json

import re
from collections import defaultdict
from pathlib import Path


HERE = Path(__file__).resolve().parent

def load(name):
    return json.loads((HERE/name).read_text(encoding="utf-8"))

def calculate(state):
    guests,nights,room,linen,service,booking=state
    return guests*(nights*room+linen) + nights*service + booking

def main():
    config=load("config.json")
    for filename,digest in config["locked_sha256"].items():
        assert hashlib.sha256((HERE/filename).read_bytes()).hexdigest() == digest
    data={}
    for filename in ("cpu_rows.jsonl","direct_rows.jsonl"):
        for line in (HERE/filename).read_text(encoding="utf8").splitlines():
            row=json.loads(line)
            key=row["pair"],row["direction"],row["step"]
            method_rows=data.setdefault(row["method"],{})
            assert key not in method_rows
            method_rows[key]=row
    cases=load("cases.json")
    assert len(cases)==48
    for pair in range(24):
        a,b=cases[2*pair:2*pair+2]
        assert a["pair"]==b["pair"]==pair and a["initial"]==b["initial"]
        assert a["turns"][:-1]==b["turns"][:-1] and a["turns"][-1]["field"]==b["turns"][-1]["field"]
    unique=set()
    for case in cases:
        state=list(map(int,re.findall(r"\d+",case["initial"])))
        for step,u in enumerate(case["turns"],1):
            index=["guests","nights"].index(u["field"])
            previous=state[index]
            state[index]=u["value"]
            assert state[index]!=previous
            if step==len(case["turns"]):
                assert (state[index]-previous)*case["direction"]>0
            total=calculate(state)
            key=(case["pair"],case["direction"],step)
            for rows in data.values():
                row=rows[key]
                assert row["id"]==u["id"] and row["truth"]==total and row["correct"]==(row["prediction"]==total)
                assert row["ns"]>0
            pred=data["learned_local"][key]["prediction"]
            assert data["symbolic_local"][key]["prediction"]==data["learned_rebuild"][key]["prediction"]==pred
            unique.add(u["id"])
    assert all(len(rs)==192 for rs in data.values())
    out={"independently_verified_steps_per_method":192,"rebuilt_oracle":"guests*(nights*room+linen)+nights*service+booking",
         "unique_customer_turns":len(unique),"unique_dialogues":len({u.split('/')[0] for u in unique}),"methods":{}}
    for method,rows in data.items():
        strata={}
        for length in (3,4,5):
            sub=[r for r in rows.values() if r["length"]==length]
            seq=defaultdict(list)
            for r in sub:seq[r["pair"],r["direction"]].append(r["correct"])
            strata[length]={"steps":len(sub),"correct_steps":sum(r["correct"] for r in sub),"sequences":len(seq),"all_correct":sum(all(v) for v in seq.values())}
        out["methods"][method]={"by_length":strata,
                                 "numeric_predictions":sum(r["prediction"] is not None for r in rows.values())}
        if method != "qwen_direct":
            out["methods"][method]["wrong_during_field_error"]=sum(not r["correct"] and not r["field_correct"] for r in rows.values())
            out["methods"][method]["wrong_despite_correct_current_field"]=sum(not r["correct"] and r["field_correct"] for r in rows.values())
    (HERE/"verified.json").write_text(json.dumps(out,indent=2),encoding="utf8")
    print(json.dumps(out,indent=2))

if __name__=="__main__":
    main()


