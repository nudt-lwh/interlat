"""Swap only the canonical digit vectors and replay the frozen arithmetic edits."""
from pathlib import Path
import hashlib
import json
import sys
from unittest.mock import patch

import numpy as np
from safetensors import safe_open
from tokenizers import Tokenizer

out = Path(__file__).resolve().parent
old = out.parent / "2026-11-19_方法迭代"
sys.path.insert(0, str(old))
import iterate

def embeddings():
    directory = out.parents[1] / "artifacts/Qwen3-1.7B"
    tokenizer = Tokenizer.from_file(str(directory/"tokenizer.json"))
    key = "model.embed_tokens.weight"
    shard = json.loads((directory/"model.safetensors.index.json").read_text())["weight_map"][key]
    with safe_open(directory/shard, framework="pt", device="cpu") as f:
        table = f.get_slice(key)
        texts = [tokenizer.encode(f"s = {d}",add_special_tokens=False).ids for d in range(10)]
        return np.stack([np.mean([table[i:i+1].float().numpy()[0] for i in ids],axis=0) for ids in texts]),texts

def main():
    source = old/"formal_rows_42.jsonl"
    qwen=np.load(old/"inputs.npz")
    raw,ids=embeddings()
    variants=dict(raw_token_mean=raw,one_hot=np.eye(10),random_gaussian=np.random.default_rng(1903).standard_normal((10,256)))
    cfg=dict(fit_seed=42,random_feature_seed=1903,raw_embedding_token_ids=ids,
             source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
             baseline="archived Qwen scores", canonical_role="s",dimensions={k:v.shape[1] for k,v in variants.items()})
    (out/"config.json").write_text(json.dumps(cfg,indent=2))
    models={}
    for name,vectors in variants.items():
        features={"names":qwen["names"],"z":np.broadcast_to(vectors,(len(qwen["names"]),*vectors.shape))}
        with patch.object(iterate.np,"load",return_value=features):
            models[name]=iterate.Candidate(42,True,True)
    results={name:dict(n=0,correct=0,preserved=0,agreement=0) for name in ["qwen",*models]}
    with (out/"rows.jsonl").open("w") as rows:
        for item,line in enumerate(source.read_text().splitlines()):
            c=json.loads(line)
            nodes=c["graph"];target=c["target"];w=len(c["after"])
            pred={"qwen":[c["pred"],c["rebuild"],c["preserved"]]}
            for name,m in models.items():
                def encode(vals):
                    return [m.encode("s",v) for v in vals]
                initial=iterate.build(encode(c["inputs"]),nodes,m.execute)
                state=initial.copy();state[target]=m.encode("s",c["after"][target]);dirty={target}
                for i,(op,l,r) in enumerate(nodes,w):
                    if l in dirty or r in dirty:
                        state[i]=m.execute(op,state[l],state[r]);dirty.add(i)
                rebuilt=iterate.build(encode(c["after"]),nodes,m.execute)
                pred[name]=[round(state[-1]),round(rebuilt[-1]),all(a==b for j,(a,b) in enumerate(zip(initial,state)) if j not in dirty)]
            for name,(local,rebuilt,kept) in pred.items():
                d=results[name];d["n"]+=1;d["correct"]+=local==c["truth"];d["agreement"]+=local==rebuilt;d["preserved"]+=kept
            rows.write(json.dumps(dict(source_line=item+1,gold=c["truth"],pred_rebuild_preserved=pred))+"\n")
    (out/"summary.json").write_text(json.dumps(results,indent=2))
    print(results)
if __name__=="__main__":
    main()
