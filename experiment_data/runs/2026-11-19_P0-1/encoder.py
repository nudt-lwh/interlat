"""Frozen Qwen forward as in extract.py, including encoder adapter and projected mean pooling."""
from pathlib import Path
import json
import sys
import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer

here = Path(__file__).resolve().parent
runs = here.parent
method = runs / "2026-11-19_方法迭代"
root = runs.parents[1]
sys.path.insert(0, str(method))
from iterate import Candidate, build

def formal_rows():
    with (method / "formal_rows_42.jsonl").open() as f:
        return [json.loads(line) for line in f]

class FrozenEncoder:
    def __init__(self, device):
        self.device = device
        base = root / "experiment_data/artifacts/Qwen3-1.7B"
        ck = torch.load(runs / "2026-11-19_云实验/fork_best.pt", map_location="cpu", weights_only=True)["model"]
        self.lm = AutoModelForCausalLM.from_pretrained(base, dtype=torch.bfloat16).eval()
        self.lm.model.layers.float().to(device)
        self.lm.model.norm.float().to(device)
        self.lm.model.rotary_emb.to(device)
        for name, layer in self.lm.named_modules():
            if name.endswith((".q_proj", ".v_proj")):
                prefix = "backbone.base_model.model." + name + ".lora_"
                a = ck[prefix+"A.encoder.weight"].bfloat16().float().to(device)
                b = ck[prefix+"B.encoder.weight"].bfloat16().float().to(device)
                layer.register_forward_hook(lambda m,x,y,a=a,b=b: y+2*((x[0]@a.T)@b.T))
        self.enc = torch.nn.Linear(2048,256)
        self.enc.load_state_dict({k[9:]:v for k,v in ck.items() if k.startswith("enc_proj.")})
        self.enc.to(device).eval()
        self.tok = AutoTokenizer.from_pretrained(base)

    @torch.inference_mode()
    def __call__(self, texts):
        t = self.tok(texts, padding=True, return_tensors="pt")
        emb = self.lm.get_input_embeddings()(t.input_ids).float().to(self.device)
        m = t.attention_mask.to(self.device)
        h = self.lm.model(inputs_embeds=emb, attention_mask=m, use_cache=False).last_hidden_state
        return self.enc((h*m[:,:,None]).sum(1)/m.sum(1)[:,None]).cpu().numpy().astype(float)

def number(model, digits):
    result = digits[0]
    a,b,c = model.place
    for digit in digits[1:]:
        result = a+b*result+c*digit
    return result

def edit(model, state, target, value, graph, n_inputs):
    local = state.copy()
    local[target] = value
    dirty = {target}
    for i,(op,l,r) in enumerate(graph, n_inputs):
        if l in dirty or r in dirty:
            local[i] = model.execute(op,local[l],local[r])
            dirty.add(i)
    return local,dirty
