"""Extract node features and independently read the previous fork checkpoint on dev."""
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

out=Path(__file__).resolve().parent
project=out.parents[2]
previous=out.parent/"neural_baselines"
sys.path.extend([str(previous),str(project)])
from branch_data import reference, schedule, splits
from interlat.models.workspace_model import Executor,StateHead,decode_digits


def restore(module, weights, prefix):
    module.load_state_dict({k[len(prefix):]:v for k,v in weights.items() if k.startswith(prefix)})
    return module.cuda().eval()


@torch.inference_mode()
def main():
    base=project/"experiment_data/artifacts/Qwen3-1.7B"
    ck=torch.load(previous/"fork_best.pt",map_location="cpu",weights_only=True)["model"]
    lm=AutoModelForCausalLM.from_pretrained(base,dtype=torch.bfloat16).eval()
    # Keep the large vocabulary matrix on CPU. Blocks use the archived run's fp32 arithmetic.
    lm.model.layers.float().cuda()
    lm.model.norm.float().cuda()
    lm.model.rotary_emb.cuda()
    tok=AutoTokenizer.from_pretrained(base)
    hooks=[]
    def adapter(name):
        while hooks: hooks.pop().remove()
        for n,layer in lm.named_modules():
            if not n.endswith((".q_proj",".v_proj")): continue
            prefix="backbone.base_model.model."+n+".lora_"
            a=ck[prefix+"A."+name+".weight"].bfloat16().float().cuda()
            b=ck[prefix+"B."+name+".weight"].bfloat16().float().cuda()
            hooks.append(layer.register_forward_hook(lambda m,x,y,a=a,b=b: y+2*((x[0]@a.T)@b.T)))
    adapter("encoder")
    enc=restore(torch.nn.Linear(2048,256),ck,"enc_proj.")
    read=restore(torch.nn.Linear(512,2048),ck,"read_proj.")
    exe=restore(Executor(256,4,node_tokens=2),ck,"executor.")
    head=restore(StateHead(256,node_tokens=2),ck,"state_head.")
    names=["s"]+[f"x{i}" for i in range(16)]
    raw=[f"{n} = {v}" for n in names for v in range(10)]
    rep=[]
    for i in range(0,len(raw),12):
        t=tok(raw[i:i+12],padding=True,return_tensors="pt").to("cuda")
        emb=lm.get_input_embeddings()(t.input_ids.cpu()).float().cuda()
        h=lm.model(inputs_embeds=emb,attention_mask=t.attention_mask,use_cache=False).last_hidden_state
        m=t.attention_mask.unsqueeze(-1)
        rep.append(enc((h*m).sum(1)/m.sum(1)))
    rep=torch.cat(rep).reshape(len(names),10,256)
    np.savez(out/"inputs.npz",z=rep.cpu().numpy(),names=names)
    adapter("reader")
    results={}
    for w in (2,4,8):
        ns,ops=schedule(w)
        dev=splits(w)[1]
        counts=defaultdict(lambda:[0,0])
        for i in range(0,len(dev),12):
            vs=dev[i:i+12]; b=len(vs)
            vals=torch.tensor(vs,device="cuda")
            expected=[reference(w,v) for v in vs]
            z={n:rep[names.index(n),vals[:,j]][:,None,:].repeat(1,2,1) for j,n in enumerate(ns)}
            depths=dict.fromkeys(ns,0)
            for n,_,(l,r) in ops:
                depths[n]=1+max(depths[l],depths[r])
                z[n]=exe(torch.stack([z[l],z[r]],1),torch.ones(b,dtype=torch.long,device="cuda"),
                         torch.tensor([0,1],device="cuda").expand(b,2))
            for n,h in z.items():
                sign,digits,_=head(h)
                preds=[("-" if s else "")+decode_digits(d) for s,d in zip(sign.argmax(-1).tolist(),digits.argmax(-1).tolist())]
                row=counts[f"depth{depths[n]}"]
                row[0]+=sum(p==str(e[n]) for p,e in zip(preds,expected));row[1]+=b
            t=tok(["result? Output nodes: g"]*b,return_tensors="pt").to("cuda")
            embed=torch.cat([lm.get_input_embeddings()(t.input_ids.cpu()).float().cuda(),read(z["g"].flatten(1))[:,None,:]],1)
            ids=[]
            done=torch.zeros(b,dtype=torch.bool)
            for step in range(8):
                last=lm.model(inputs_embeds=embed,use_cache=False).last_hidden_state[:,-1].cpu()
                scores=torch.cat([last@part.float().T for part in lm.lm_head.weight.split(8192)],dim=1)
                token=scores.argmax(1).masked_fill(done,tok.eos_token_id)
                ids.append(token)
                done|=token==tok.eos_token_id
                if done.all():break
                embed=torch.cat([embed,lm.get_input_embeddings()(token[:,None]).float().cuda()],1)
            ids=torch.stack(ids,1)
            row=counts["reader"];row[0]+=sum(p.strip()==str(e["g"]) for p,e in zip(tok.batch_decode(ids,skip_special_tokens=True),expected));row[1]+=b
        results[w]=dict(counts);print(w,dict(counts),flush=True)
    (out/"diagnosis.json").write_text(json.dumps(results,indent=2))


if __name__=="__main__":
    torch.set_num_threads(4)
    main()
