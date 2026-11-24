"""python -B prepare.py && python -B evaluate.py && python -B verify.py"""
import sys
sys.dont_write_bytecode=True
from prepare import *
import torch
import transformers
from transformers import AutoModelForCausalLM,AutoTokenizer
sys.path.insert(0,str(folder.parent/"2026-11-19_P0-2"))
from shared import frozen_model

def apply_choice(workspace,row,pred):
    qty,price,fee=[workspace.encode("s",row[k]) for k in ("qty","price","fee")]
    _,field,value,_=row["proposals"][pred]
    subtotal=workspace.execute(2,qty,price)
    if field=="unit price":
        subtotal=workspace.execute(2,qty,workspace.encode("s",value))
    else:
        fee=workspace.encode("s",value)
    return round(workspace.execute(0,subtotal,fee))

def run():
    cfg=json.loads((folder/"config.json").read_text())
    for p,key in [(Path(__file__),"eval_script_sha256"),(test,"test_sha256"),(folder/"demos.json","demos_sha256"),(source/"calibration_42.npz","calibration_sha256"),(base/"model.safetensors.index.json","model_index_sha256")]:
        assert digest(p)==cfg[key]
    data=load_data(test)
    demos=json.loads((folder/"demos.json").read_text(encoding="utf-8"))
    tok=AutoTokenizer.from_pretrained(base,padding_side="left",local_files_only=True)
    lm=AutoModelForCausalLM.from_pretrained(base,dtype=torch.bfloat16,local_files_only=True).cuda().eval()
    letters=[tok.encode(c,add_special_tokens=False) for c in "AB"]
    assert all(len(x)==1 for x in letters)
    ids=[x[0] for x in letters]
    prefix=[]
    for r in demos:
        prefix.extend([dict(role="user",content=r["text"]),dict(role="assistant",content=r["answer"])])
    typed=frozen_model()
    save("runtime.json",dict(device=torch.cuda.get_device_name(),torch=torch.__version__,transformers=transformers.__version__,label_token_ids=ids))
    with torch.inference_mode(),(folder/"rows.jsonl").open("w",encoding="utf-8") as of:
        for i in range(0,len(data),cfg["batch_size"]):
            batch=data[i:i+cfg["batch_size"]]
            texts=[tok.apply_chat_template(prefix+[dict(role="user",content=r["text"])],tokenize=False,add_generation_prompt=True,enable_thinking=False) for r in batch]
            t=tok(texts,padding=True,return_tensors="pt").to("cuda")
            logits=lm(**t,logits_to_keep=1,use_cache=False).logits[:,-1,ids].float().cpu()
            for j,(r,s,pred) in enumerate(zip(batch,logits.tolist(),logits.argmax(-1).tolist())):
                z=dict(source_line=i+j+1,group=r["group"],pair=r["pair"],variant=r["variant"],text=r["text"],gold=r["answer"],
                    pred=pred,gold_total=r["total"],pred_total=apply_choice(typed,r,pred),zero_pred=r["direct_choice"],
                    grounded_pred=r["pred"],logit_A=s[0],logit_B=s[1],tokens=int(t.attention_mask[j].sum()))
                of.write(json.dumps(z)+"\n")
            of.flush()
            if i%80==0:print("scored",i+len(batch),flush=True)

if __name__=="__main__":
    run()
