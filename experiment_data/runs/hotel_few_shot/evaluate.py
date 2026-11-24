"""Frozen four-shot hotel baseline. Run at batch size one on the same 192 requests."""
import hashlib
import json
import platform
import re
import sys
from pathlib import Path
from statistics import median
from time import perf_counter_ns

import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer

HERE=Path(__file__).resolve().parent
P1=HERE.parent/"hotel_replay"
sys.path.insert(0,str(P1))
from run import FIELDS, dump, initial_values, oracle, prompt, read_json

def main():
    cfg=read_json(HERE/"config.json")
    for name,digest in cfg["locked_sha256"].items():
        assert hashlib.sha256((HERE/name).read_bytes()).hexdigest()==digest,name
    demos=read_json(HERE/"demos.json")
    cases=read_json(P1/"cases.json")
    demo_ids={u["id"].split("/")[0] for d in demos for u in d["turns"]}
    test_ids={u["id"].split("/")[0] for c in cases for u in c["turns"]}
    assert not demo_ids & test_ids
    prefix=[message for d in demos for message in d["messages"]]
    base=HERE.parents[1]/"artifacts/Qwen3-1.7B"
    tok=AutoTokenizer.from_pretrained(base,local_files_only=True)
    model=AutoModelForCausalLM.from_pretrained(base,dtype=torch.bfloat16,local_files_only=True).to("cuda").eval()

    @torch.inference_mode()
    def answer(desc, history):
        messages=[*prefix,dict(role="user",content=prompt(desc,history))]
        text=tok.apply_chat_template(messages,tokenize=False,add_generation_prompt=True,enable_thinking=False)
        t=tok(text,return_tensors="pt").to("cuda")
        generated=model.generate(**t,do_sample=False,max_new_tokens=16,pad_token_id=tok.eos_token_id)
        output=tok.decode(generated[0,t.input_ids.shape[1]:],skip_special_tokens=True).strip()
        m=re.fullmatch(r"\$?\s*([0-9][0-9,]*)(?:\.0+)?\s*(?:dollars?|USD)?[.!]?",output,re.I)
        return dict(raw=output,prediction=int(m[1].replace(",","")) if m else None,
                    input_tokens=t.input_ids.shape[1],output_tokens=generated.shape[1]-t.input_ids.shape[1])
    answer("Guests: 2. Nights: 3. Rate: 1 dollar per guest per night.",[])
    rows=[]
    for c in cases:
        state=initial_values(c["initial"])
        for k,u in enumerate(c["turns"],1):
            torch.cuda.synchronize()
            start=perf_counter_ns()
            output=answer(c["initial"],c["turns"][:k])
            torch.cuda.synchronize()
            elapsed=perf_counter_ns()-start
            state[FIELDS.index(u["field"])]=u["value"]
            gold=oracle(state)
            rows.append(dict(pair=c["pair"],direction=c["direction"],step=k,length=len(c["turns"]),id=u["id"],
                             **output,truth=gold,correct=output["prediction"]==gold,ns=elapsed))
        print(c["pair"],c["direction"],flush=True)
    (HERE/"rows.jsonl").write_text("".join(json.dumps(r)+"\n" for r in rows),encoding="utf8")
    seq={(c["pair"],c["direction"]):[r for r in rows if r["pair"]==c["pair"] and r["direction"]==c["direction"]] for c in cases}
    last={key:v[-1]["correct"] for key,v in seq.items()}
    summary=dict(steps=len(rows),numeric=sum(r["prediction"] is not None for r in rows),
        correct_steps=sum(r["correct"] for r in rows),sequences=len(seq),
        all_steps_correct=sum(all(r["correct"] for r in q) for q in seq.values()),
        final_correct=sum(last.values()),pairs=len(seq)//2,
        both_finals_correct=sum(last[p,-1] and last[p,1] for p,d in last if d==-1),
        min_prompt_tokens=min(r["input_tokens"] for r in rows),max_prompt_tokens=max(r["input_tokens"] for r in rows),
        median_step_ms=median(r["ns"] for r in rows)/1e6, demo_dialogues=len(demo_ids),
        test_dialogues=len(test_ids),overlapping_dialogues=len(test_ids & demo_ids),
        torch=torch.__version__,transformers=transformers.__version__,python=platform.python_version(),
        gpu=torch.cuda.get_device_name())
    dump(HERE/"summary.json",summary)

if __name__=="__main__":main()
