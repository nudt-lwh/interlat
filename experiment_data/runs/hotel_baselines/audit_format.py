"""Post-hoc final-line format sensitivity. Extraction takes only raw model output."""
import json
import re
from collections import Counter,defaultdict
from pathlib import Path
import e8
from verify_e8 import verify

HERE=Path(__file__).resolve().parent
ROOT=HERE/"results"

def extract(output):
    strict=e8.amount(output)
    if strict is not None:
        return strict,"strict"
    lines=output.strip().splitlines()
    if not lines:
        return None,"no_final_integer"
    line=lines[-1].strip()
    # Only two syntactic relaxations; no lookup in preceding reasoning, oracle, or other rows.
    heading=re.sub(r"^#{1,6}\s+","",line)
    numeric=e8.amount(heading)
    if numeric is not None:
        return numeric,"markdown_heading"
    currency=re.sub(r"^total\s+in\s+whole\s+dollars\s*:\s*","",heading.replace("*",""),flags=re.I)
    numeric=e8.amount(currency)
    return (numeric,"explicit_total_in_whole_dollars") if numeric is not None else (None,"no_final_integer")

def main():
    summed={mode:verify(ROOT/mode) for mode in e8.MODES}
    e8.write(HERE/"summary.json",summed)
    combined=[]
    diagnostic={}
    audit=[]
    for mode in e8.MODES:
        rows=[json.loads(l) for l in (ROOT/mode/"rows.jsonl").read_text(encoding="utf8").splitlines()]
        combined.extend([dict(mode=mode,**r) for r in rows])
        if mode=="frontend":continue
        parsed=[extract(r["output"]) for r in rows]
        seq=defaultdict(list)
        count=Counter()
        for r,(pred,reason) in zip(rows,parsed):
            correct=pred==r["truth"]
            seq[tuple(r["key"][:2])].append(correct)
            count[reason]+=1
            audit.append(dict(mode=mode,key=r["key"],category=reason,extracted=pred,correct=correct,
                              strict_correct=r["correct"],truncated=r["truncated"]))
        diagnostic[mode]=dict(categories=dict(count),parsed=sum(p is not None for p,_ in parsed),
            steps_correct=sum(sum(v) for v in seq.values()),all_steps_correct=sum(all(v) for v in seq.values()),
            final_correct=sum(v[-1] for v in seq.values()),both_finals_correct=sum(seq[i,-1][-1] and seq[i,1][-1] for i in range(24)))
    assert len(combined)==960
    (HERE/"rows.jsonl").write_text("".join(json.dumps(r,ensure_ascii=False)+"\n" for r in combined),encoding="utf8")
    folder=HERE/"posthoc_format"
    folder.mkdir(exist_ok=True)
    e8.write(folder/"summary.json",dict(protocol="Post-hoc sensitivity after inspecting model output formatting; no new inference. Each mode is rescored with the identical rule. Read only the final nonempty line, strip leading Markdown heading and allow the explicit label 'Total in whole dollars:'. Never select among candidates by truth. Retain main scores unchanged.",results=diagnostic))
    (folder/"rows.jsonl").write_text("".join(json.dumps(r)+"\n" for r in audit),encoding="utf8")
    print(json.dumps(diagnostic,indent=2))
if __name__=="__main__":
    main()
