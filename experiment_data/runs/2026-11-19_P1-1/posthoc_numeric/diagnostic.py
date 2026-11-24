"""Post-hoc numeric-output diagnostic; the main zero-shot protocol is retained."""
import hashlib
import json
import sys
from pathlib import Path
from time import perf_counter_ns
from statistics import median
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, LogitsProcessor, LogitsProcessorList

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent))
from run import read_json, dump, prompt, initial_values, oracle, FIELDS

class UnsignedInteger(LogitsProcessor):
    """Accept one to nine digits, then end; canonical integers have no leading zero."""
    def __init__(self, tok):
        self.ids=[tok.encode(str(k),add_special_tokens=False)[0] for k in range(10)]
        assert all(tok.encode(str(k),add_special_tokens=False)==[token] for k,token in enumerate(self.ids))
        self.end=tok.eos_token_id
        self.start=None

    def __call__(self, input_ids, scores):
        if self.start is None:
            self.start=input_ids.shape[1]
        length=input_ids.shape[1]-self.start
        if not length:
            allowed=self.ids
        elif length == 9 or input_ids[0,self.start].item()==self.ids[0]:
            allowed=[self.end]
        else:
            allowed=self.ids+[self.end]
        masked=torch.full_like(scores,-torch.inf)
        masked[:,allowed]=scores[:,allowed]
        return masked

def main():
    config=read_json(HERE/"config.json")
    for f,digest in config["locked_sha256"].items():
        assert hashlib.sha256((HERE/f).read_bytes()).hexdigest()==digest
    cases=read_json(HERE.parent/"cases.json")
    base=HERE.parents[2]/"artifacts/Qwen3-1.7B"
    tok=AutoTokenizer.from_pretrained(base,local_files_only=True)
    model=AutoModelForCausalLM.from_pretrained(base,dtype=torch.bfloat16,local_files_only=True).to("cuda").eval()
    @torch.inference_mode()
    def answer(desc,history):
        text=tok.apply_chat_template([dict(role="user",content=prompt(desc,history))],tokenize=False,
                                     add_generation_prompt=True,enable_thinking=False)
        inputs=tok(text,return_tensors="pt").to("cuda")
        processor=UnsignedInteger(tok)
        result=model.generate(**inputs,do_sample=False,max_new_tokens=10,pad_token_id=tok.eos_token_id,
                              logits_processor=LogitsProcessorList([processor]))
        raw=tok.decode(result[0,inputs.input_ids.shape[1]:],skip_special_tokens=True)
        assert raw.isascii() and raw.isdigit() and 1<=len(raw)<=9
        return raw
    answer("Guests: 2. Nights: 3. Rate: 1 dollar per guest per night.",[])
    rows=[]
    for c in cases:
        state=initial_values(c["initial"])
        for k,u in enumerate(c["turns"],1):
            torch.cuda.synchronize()
            start=perf_counter_ns()
            raw=answer(c["initial"],c["turns"][:k])
            torch.cuda.synchronize()
            elapsed=perf_counter_ns()-start
            state[FIELDS.index(u["field"])]=u["value"]
            rows.append(dict(pair=c["pair"],direction=c["direction"],step=k,id=u["id"],raw=raw,
                             truth=oracle(state),correct=int(raw)==oracle(state),ns=elapsed))
    (HERE/"rows.jsonl").write_text("".join(json.dumps(r)+"\n" for r in rows),encoding="utf8")
    sequences=[[r for r in rows if r["pair"]==c["pair"] and r["direction"]==c["direction"]] for c in cases]
    dump(HERE/"summary.json",dict(n=len(rows),numeric=len(rows),correct=sum(r["correct"] for r in rows),
          sequences=len(sequences),all_correct=sum(all(r["correct"] for r in seq) for seq in sequences),
          final_correct=sum(seq[-1]["correct"] for seq in sequences),median_step_ms=median(r["ns"] for r in rows)/1e6))
if __name__=="__main__":main()
