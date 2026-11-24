"""Lock six balanced development examples before scoring the fixed test corpus."""
import hashlib
import json
import random
from pathlib import Path

folder=Path(__file__).resolve().parent
source=folder.parent/"arithmetic_workspace"
base=folder.parents[1]/"artifacts"/"Qwen3-1.7B"
dev=source/"procurement_dev_grounded.jsonl"
test=source/"procurement_formal_grounded.jsonl"

def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def load_data(path):
    return [json.loads(line) for line in path.open(encoding="utf-8")]

def save(name,value):
    (folder/name).write_text(json.dumps(value,indent=2,ensure_ascii=False),encoding="utf-8")

def select():
    rng=random.Random(20261120)
    rows=load_data(dev)
    selected=[]
    for g in range(3):
        for v in range(2):
            answer=(g+v)%2
            options=[(i,r) for i,r in enumerate(rows,1) if (r["group"],r["variant"],r["answer"])==(g,v,answer)]
            i,r=rng.choice(options)
            selected.append(dict(dev_line=i,group=g,variant=v,pair=r["pair"],text=r["text"],answer="AB"[answer]))
    rng.shuffle(selected)
    return selected

if __name__=="__main__":
    demos=select()
    save("demos.json",demos)
    formal=load_data(test)
    assert not {r["text"] for r in demos}&{r["text"] for r in formal}
    assert not {p[0] for r in load_data(dev) for p in r["proposals"]}&{p[0] for r in formal for p in r["proposals"]}
    assert len({(r["group"],r["pair"]) for r in demos})==6
    save("config.json",dict(model="local unadapted Qwen3-1.7B",model_dir="experiment_data/artifacts/Qwen3-1.7B",
        source_dev=dev.name,source_test=test.name,seed=20261120,demonstrations=6,
        selection="Fixed one development example for each group and variant (3 x 2); target label A/B alternates by group+variant. Fixed random within eligible stratum, then shuffled. Distinct development suppliers and decision templates. No formal outcomes used in selection.",
        protocol="Six answered chat user/assistant exchanges followed by original unmodified test user turn. Same original batch order, letter logits, bf16, left padding, non-thinking chat template, and typed amount executor. No rule fallback; frozen archived direct_choice is the paired zero-shot baseline.",
        batch_size=8,eval_script_sha256=digest(folder/"evaluate.py"),original_script_sha256=digest(source/"procurement.py"),
        dev_sha256=digest(dev),test_sha256=digest(test),demos_sha256=digest(folder/"demos.json"),
        calibration_sha256=digest(source/"calibration_42.npz"),model_index_sha256=digest(base/"model.safetensors.index.json")))
