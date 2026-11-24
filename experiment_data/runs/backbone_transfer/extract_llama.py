"""Extract the ten canonical digit vectors on the project cloud GPU."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("project", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    base = args.project / "experiment_data/artifacts/Llama-3.2-1B"
    checkpoint = args.project / "experiment_data/runs/llama_s2_mul/ckpt_final.pt"
    weights = torch.load(checkpoint, map_location="cpu", weights_only=False)
    weights = weights.get("model", weights)
    model = AutoModelForCausalLM.from_pretrained(base, torch_dtype=torch.bfloat16).float().cuda().eval()
    tok = AutoTokenizer.from_pretrained(base)
    tok.pad_token = tok.eos_token
    projection = torch.nn.Linear(model.config.hidden_size, weights["enc_proj.bias"].numel()).cuda().eval()
    projection.load_state_dict({n.removeprefix("enc_proj."): v for n,v in weights.items() if n.startswith("enc_proj.")})
    count = 0
    for name, layer in model.named_modules():
        if not name.endswith((".q_proj", ".v_proj")):
            continue
        prefix = "backbone.base_model.model." + name + ".lora_"
        a = weights[prefix + "A.encoder.weight"].bfloat16().float().cuda()
        b = weights[prefix + "B.encoder.weight"].bfloat16().float().cuda()
        layer.register_forward_hook(lambda m,x,y,a=a,b=b: y+2*((x[0]@a.T)@b.T))
        count += 1
    assert count == 2*model.config.num_hidden_layers
    with torch.inference_mode():
        batch = tok([f"s = {d}" for d in range(10)], padding=True,return_tensors="pt").to("cuda")
        h = model.model(**batch, use_cache=False).last_hidden_state
        mask = batch.attention_mask.unsqueeze(-1)
        vectors = projection((h*mask).sum(1)/mask.sum(1)).reshape(10,-1,256).mean(1).cpu().numpy()
    assert vectors.shape == (10,256) and np.isfinite(vectors).all()
    args.output.mkdir(parents=True,exist_ok=True)
    np.savez(args.output/"inputs.npz",names=["s"],z=vectors[None])
    print(json.dumps(dict(model="Llama-3.2-1B-Instruct", adapter_modules=count, gpu=torch.cuda.get_device_name(), checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(), features_sha256=hashlib.sha256((args.output/"inputs.npz").read_bytes()).hexdigest())))
if __name__=="__main__":
    main()
