"""Frozen E8 hotel replay protocols. All paths are relative to this file or CLI arguments."""
import argparse
import hashlib
import json
import re
from collections import defaultdict
from pathlib import Path
from time import perf_counter

import numpy as np
from iterate import Candidate, build

HOME_DIR = Path(__file__).resolve().parent
MODES = ("small_cot0", "small_cot4", "large_direct", "large_cot", "frontend")
FIELDS = ["guests", "nights"]
GRAPH = [(2,0,1),(2,6,2),(2,0,3),(2,1,4),(0,7,8),(0,10,9),(0,11,5)]
INTRO = ("Reprice this hotel booking. Apply each customer's new guest count or stay length in order. "
         "Each exchange is a separate amendment. Ignore unrelated requests. Use the clerk's context "
         "only to interpret that customer's words. Keep other bill terms fixed.")
REASON = ("Show your work. Identify each amendment and the resulting number of guests and nights. "
          "Calculate all charges and add them. End on a separate final line with: Final: <total in whole dollars>.")
DIRECT = "What is the updated total cost in dollars? Output only the whole number."
FRONT = ('In this hotel booking exchange, identify the amendment the customer requests. '
         'Use the clerk as context. Only guest count and stay length can change. '
         'Return just one JSON object: {"field":"guests","new_value":3} or '
         '{"field":"nights","new_value":2}, replacing the example with the actual field and integer. '
         'If neither is identifiable, return null.')

def read(path):
    return json.loads(path.read_text(encoding="utf8"))

def write(path, data):
    staging=path.with_suffix(".tmp")
    staging.write_text(json.dumps(data,ensure_ascii=False,indent=2),encoding="utf8")
    staging.replace(path)

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def numbers(desc):
    return [int(s) for s in re.findall(r"\d+",desc)]

def gold_total(values):
    g,n,r,l,s,b=values
    return g*n*r+g*l+n*s+b

def exchange(turn):
    return f"Clerk context: {turn['context']}\nCustomer: {turn['text']}"

def bill_prompt(desc,turns,ending):
    return "\n".join([INTRO,desc,*map(exchange,turns),ending])

def demonstrations():
    chats=[]
    for demo in read(HOME_DIR/"demos.json"):
        values=numbers(demo["initial"])
        explanation=[]
        for u in demo["turns"]:
            values[FIELDS.index(u["field"])]=u["value"]
            explanation.append(f"Set {u['field']} to {u['value']}.")
        g,n,r,l,s,b=values
        amount=gold_total(values)
        assert amount==demo["gold"]
        explanation.extend([f"The bill now covers {g} guests for {n} nights.",
             f"Room = {g} x {n} x {r} = {g*n*r}. Linen = {g} x {l} = {g*l}. "
             f"Service = {n} x {s} = {n*s}. Booking = {b}.",
             f"Total = {g*n*r} + {g*l} + {n*s} + {b} = {amount}.", f"Final: {amount}"])
        chats.extend([{"role":"user","content":bill_prompt(demo["initial"],demo["turns"],REASON)},
                      {"role":"assistant","content":"\n".join(explanation)}])
    return chats

def tasks(cases,mode):
    demos=demonstrations() if mode=="small_cot4" else []
    for c in cases:
        for step,u in enumerate(c["turns"],1):
            if mode=="frontend":
                msg=FRONT+"\n"+exchange(u)
            else:
                msg=bill_prompt(c["initial"],c["turns"][:step],DIRECT if mode=="large_direct" else REASON)
            yield (c["pair"],c["direction"],step),[*demos,{"role":"user","content":msg}]

def parse_pair(raw):
    text=raw.strip()
    if text.startswith("```"):
        text=re.sub(r"^```(?:json)?\s*|\s*```$","",text,flags=re.I)
    try:
        parsed=json.loads(text)
    except (ValueError,TypeError):
        return None
    if not isinstance(parsed,dict) or parsed.get("field") not in FIELDS:
        return None
    v=parsed.get("new_value")
    if type(v) is not int or not 1<=v<=999999:
        return None
    return parsed["field"],v

def amount(raw):
    # Score the final nonempty line only. A line containing intermediate arithmetic is rejected.
    lines=raw.strip().splitlines()
    if not lines:
        return None
    line=lines[-1].strip().replace("*","").replace("`","").replace("\\$","$")
    line=re.sub(r"\\boxed\{([^{}]+)\}",r"\1",line)
    m=re.fullmatch(r"(?:\s*(?:final(?:\s+(?:answer|total|bill))?|(?:the\s+)?(?:total(?:\s+cost)?|answer))\s*(?:is|:|=)?\s*)?(?:USD\s*|\$\s*)?((?:\d{1,3}(?:,\d{3})+|\d+))(?:\.00?)?\s*(?:dollars?|USD)?[.!]?\s*",line,re.I)
    return int(m[1].replace(",","")) if m else None

class Workspace:
    def __init__(self):
        model=Candidate(42,True,True)
        with np.load(HOME_DIR/"calibration_42.npz") as ck:
            model.mean=ck["mean"];model.bias=float(ck["bias"]);model.weights=ck["decoder"]
            model.operators=ck["operators"];model.place=ck["place"]
        model.decoded.clear()
        self.model=model

    def start(self, desc):
        return build([self.model.encode("s",v) for v in numbers(desc)],GRAPH,self.model.execute)

    def change(self,state,edit):
        if edit:
            field,value=edit
            index=FIELDS.index(field)
            state[index]=self.model.encode("s",value)
            dirty={index}
            for j,(op,l,r) in enumerate(GRAPH,6):
                if l in dirty or r in dirty:
                    state[j]=self.model.execute(op,state[l],state[r])
                    dirty.add(j)
        return round(state[-1])

def score(cases, records, mode, folder):
    by_id={tuple(r["key"]):r for r in records}
    assert len(records)==len(by_id)==192
    workspace=Workspace() if mode=="frontend" else None
    rows=[]
    seq=defaultdict(list)
    for c in cases:
        values=numbers(c["initial"])
        state=workspace.start(c["initial"]) if workspace else None
        for step,u in enumerate(c["turns"],1):
            key=c["pair"],c["direction"],step
            row=by_id[key].copy()
            values[FIELDS.index(u["field"])]=u["value"]
            if workspace:
                edit=parse_pair(row["output"])
                pred=workspace.change(state,edit)
                row.update(edit=edit,field_ok=bool(edit and edit[0]==u["field"]),
                           edit_ok=edit==(u["field"],u["value"]),parsed=edit is not None)
            else:
                pred=amount(row["output"])
                row["parsed"]=pred is not None
            row.update(id=u["id"],prediction=pred,truth=gold_total(values),correct=pred==gold_total(values))
            rows.append(row)
            seq[key[:2]].append(row["correct"])
    staging=folder/"rows.tmp"
    staging.write_text("".join(json.dumps(r,ensure_ascii=False)+"\n" for r in rows),encoding="utf8")
    staging.replace(folder/"rows.jsonl")
    report=dict(steps=len(rows),correct_steps=sum(r["correct"] for r in rows),
                parsed=sum(r["parsed"] for r in rows),truncated=sum(r["truncated"] for r in rows),
                sequences=len(seq),all_steps_correct=sum(all(v) for v in seq.values()),
                final_correct=sum(v[-1] for v in seq.values()),pairs=len(seq)//2,
                both_finals_correct=sum(seq[p,-1][-1] and seq[p,1][-1] for p,d in seq if d==-1))
    if workspace:
        report.update(field_correct=sum(r["field_ok"] for r in rows),edit_correct=sum(r["edit_ok"] for r in rows))
    write(folder/"summary.json",report)
    print(mode,report,flush=True)

def generate(model_path,modes,root,batch_size):
    import torch,transformers
    from transformers import AutoModelForCausalLM,AutoTokenizer
    cfg=read(HOME_DIR/"config.json")
    for name,digest in cfg["sha256"].items():
        assert sha(HOME_DIR/name)==digest,name
    cases=read(HOME_DIR/"cases.json")
    tok=AutoTokenizer.from_pretrained(model_path,local_files_only=True,padding_side="left")
    if tok.pad_token_id is None:
        tok.pad_token=tok.eos_token
    lm=AutoModelForCausalLM.from_pretrained(model_path,torch_dtype=torch.bfloat16,local_files_only=True).to("cuda").eval()
    eos=lm.generation_config.eos_token_id
    stops={eos} if isinstance(eos,int) else set(eos or [])
    stops.add(tok.eos_token_id)
    for mode in modes:
        folder=root/mode
        folder.mkdir(parents=True,exist_ok=True)
        budget=cfg["max_new_tokens"][mode]
        meta=dict(mode=mode,model=str(model_path.resolve()),model_config_hash=sha(model_path/"config.json"),
                  protocol_hash=sha(HOME_DIR/"config.json"),batch_size=batch_size,max_new_tokens=budget,
                  transformers=transformers.__version__,torch=torch.__version__,gpu=torch.cuda.get_device_name())
        if (folder/"metadata.json").exists():
            assert read(folder/"metadata.json")==meta,"Use a new output folder for a changed configuration"
        write(folder/"metadata.json",meta)
        work=list(tasks(cases,mode))
        records=[]
        for i in range(0,len(work),batch_size):
            chunk=work[i:i+batch_size]
            batchfile=folder/f"batch_{i:04d}.json"
            if batchfile.exists():
                batch=read(batchfile)
                assert [r["key"] for r in batch]==[list(k) for k,_ in chunk]
            else:
                texts=[tok.apply_chat_template(chat,tokenize=False,add_generation_prompt=True,enable_thinking=False) for _,chat in chunk]
                inputs=tok(texts,return_tensors="pt",padding=True,add_special_tokens=False).to("cuda")
                torch.cuda.synchronize()
                start=perf_counter()
                with torch.inference_mode():
                    gen=lm.generate(**inputs,do_sample=False,max_new_tokens=budget,pad_token_id=tok.pad_token_id)
                torch.cuda.synchronize()
                elapsed=perf_counter()-start
                batch=[]
                for (key,_),text,ids,mask in zip(chunk,texts,gen[:,inputs.input_ids.shape[1]:].tolist(),inputs.attention_mask.tolist()):
                    length=next((j+1 for j,k in enumerate(ids) if k in stops),len(ids))
                    ended=ids[length-1] in stops
                    raw=tok.decode(ids[:length],skip_special_tokens=True)
                    batch.append(dict(key=key,prompt_sha256=hashlib.sha256(text.encode()).hexdigest(),
                        output=raw,input_tokens=sum(mask),output_tokens=length,truncated=not ended and length>=budget,
                        batch_seconds=elapsed))
                write(batchfile,batch)
                print(mode,i+len(chunk),"/",len(work),flush=True)
            records.extend(batch)
        score(cases,records,mode,folder)

if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--model",type=Path,required=True)
    parser.add_argument("--modes",nargs="+",choices=MODES,required=True)
    parser.add_argument("--out",type=Path,required=True)
    parser.add_argument("--batch-size",type=int,default=8)
    args=parser.parse_args()
    generate(args.model,args.modes,args.out,args.batch_size)
