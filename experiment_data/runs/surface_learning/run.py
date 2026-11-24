"""Fixed sample-size probe of contextual single-digit decoding. Run prepare, extract, then evaluate."""
import argparse
from functools import lru_cache
import hashlib
import json
from pathlib import Path
import random
import re
import sys

import numpy as np

OUT = Path(__file__).resolve().parent
ORIGINAL = OUT.parent / "input_formats"
METHOD = OUT.parent / "arithmetic_workspace"
sys.path.insert(0, str(ORIGINAL))
BUDGETS = (10, 30, 100)
GROUPS = ("names", "sentence", "words")
WORDS = "zero one two three four five six seven eight nine".split()
FORMS = ["the value of {n} is {v}", "{n} has value {v}", "{n}: {v}",
         "{v} is the value of {n}", "let {n} be {v}", "{n} equals {v}",
         "set {n} to {v}", "assign {v} to {n}", "for {n}, use {v}", "{n} holds {v}"]
PRIMES = "amber blue cedar dusk elm fawn gold hazel ivory jade kelp lime moss navy oak plum reed sage teal umber".split()
NOUNS = "alder brook cove dune ember fern glen hill isle jade knoll lake mesa nook pond".split()
SEED = 194721
BATCH = 24

def save(name, data):
    (OUT/name).write_text(json.dumps(data,ensure_ascii=False,indent=2),encoding="utf-8")

def rows_for_protocol():
    vocab=[f"{stem}_{noun}" for stem in PRIMES for noun in NOUNS]
    random.Random(SEED).shuffle(vocab)
    train_names, test_names = vocab[:100],vocab[100:200]
    records=[]
    for rank,name in enumerate(train_names):
        for digit in range(10):
            records.append(dict(group="train",rank=rank+1,name=name,digit=digit,text=f"{name} = {digit}"))
    for rank,name in enumerate(test_names):
        form=FORMS[rank%len(FORMS)]
        for digit in range(10):
            for group,text in zip(GROUPS,(f"{name} = {digit}",form.format(n=name,v=digit),f"{name} = {WORDS[digit]}")):
                records.append(dict(group=group,rank=rank+1,name=name,digit=digit,text=text))
    assert len(records)==len({r["text"] for r in records})==4000
    assert set(train_names).isdisjoint(test_names)
    return records

def prepare():
    rows=rows_for_protocol()
    raw="".join(json.dumps(r)+"\n" for r in rows).encode("utf-8")
    (OUT/"stimuli.jsonl").write_bytes(raw)
    save("config.json",dict(seed=SEED,budgets_per_digit=BUDGETS,digits=list(range(10)),group_size=1000,groups=GROUPS,
         train_definition="Nested prefixes of 100 randomized variable names, each paired with every Arabic digit in name = d. All 100 held-out names differ from training names.",
         test_definition="Three paired tests use identical held-out name/digit pairs. Sentences additionally alter the template (10 unseen forms each used with 10 names, independent of digit). Words additionally replace the Arabic digit with the corresponding English word.",
         unique_sentences=FORMS,decoder="P0-1 frozen Qwen3-1.7B with existing encoder LoRA and 256d projection; affine least squares with centering and fixed rcond=1e-7, raw features, nearest integer without clipping. No test-driven selection.",
         adapter="Full-string grammar and English-digit dictionary canonicalize to s = d. Canonical supervised affine readout refit at each budget; ten unique canonical strings each repeated budget times. This control supplies grammar and lexicon.",
         endpoint="Single-digit numeric exact match on held-out assignment text; not downstream full edit accuracy. First observed budget >=95%; no interpolation. One fixed name split.",
         extract_batch=BATCH,stimuli_sha256=hashlib.sha256(raw).hexdigest(),source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()))

def extract(rows):
    from encoder import FrozenEncoder,torch
    torch.set_num_threads(4)
    enc=FrozenEncoder("cuda")
    texts=[r["text"] for r in rows]
    # Fix padding batches independently of budget. Do not use test examples during fitting.
    groups=[]
    for label in ("train",*GROUPS):
        sorted_items=sorted((r["text"],i) for i,r in enumerate(rows) if r["group"]==label)
        groups.append(sorted_items)
    features=np.empty((len(texts),256),dtype=np.float32)
    for group in groups:
        for offset in range(0,len(group),BATCH):
            batch=group[offset:offset+BATCH]
            features[[i for _,i in batch]]=enc([s for s,_ in batch])
        print("encoded",len(group),flush=True)
    canonical=enc([f"s = {k}" for k in range(10)]+["x0 = 0","x0 = 1"])[:10]
    np.savez_compressed(OUT/"features.npz",features=features,canonical=canonical)

@lru_cache(None)
def grammar(form):
    return re.compile(re.escape(form).replace(re.escape("{n}"),r"[a-z_]+").replace(re.escape("{v}"),r"(?P<token>[a-z]+|[0-9])"))

def canonical_token(text):
    for form in ["{n} = {v}",*FORMS]:
        m=grammar(form).fullmatch(text)
        if m:
            token=m["token"]
            return WORDS.index(token) if token in WORDS else int(token)
    raise ValueError(text)

def fit(x,y):
    center=x.mean(0)
    bias=y.mean()
    weights=np.linalg.lstsq(x-center,y-bias,rcond=1e-7)[0]
    return center,weights,bias

def predict(coefs,x):
    center,weights,bias=coefs
    return (x-center)@weights+bias

def evaluate(rows):
    raw=np.load(OUT/"features.npz")
    z=raw["features"].astype(float)
    canonical=raw["canonical"].astype(float)
    y=np.array([r["digit"] for r in rows])
    parsed=np.array([canonical_token(r["text"]) for r in rows])
    summaries={}
    with (OUT/"rows.jsonl").open("w",encoding="utf-8") as out:
        for n in BUDGETS:
            train=[i for i,r in enumerate(rows) if r["group"]=="train" and r["rank"]<=n]
            primary=predict(fit(z[train],y[train]),z)
            adapter=predict(fit(canonical[parsed[train]],y[train]),canonical[parsed])
            for group in GROUPS:
                test=[i for i,r in enumerate(rows) if r["group"]==group]
                count=upper=0
                for i in test:
                    pred,adapt=round(primary[i]),round(adapter[i])
                    count+=int(pred==y[i])
                    upper+=int(adapt==y[i])
                    out.write(json.dumps(dict(**rows[i],fit_per_digit=n,stimulus_line=i+1,
                        raw_prediction=float(primary[i]),prediction=pred,adapter_prediction=adapt))+"\n")
                summaries[f"{group}/{n}"]=dict(n=len(test),correct=count,accuracy=count/len(test),
                                               adapter_correct=upper,adapter_accuracy=upper/len(test))
    first={group:next((n for n in BUDGETS if summaries[f"{group}/{n}"]["accuracy"]>=.95),None) for group in GROUPS}
    save("summary.json",dict(by_group_budget=summaries,first_observed_budget_95pct=first,
         adapter_first_observed_budget_95pct={g:next((n for n in BUDGETS if summaries[f"{g}/{n}"]["adapter_accuracy"]>=.95),None) for g in GROUPS}))
    print(summaries,first,flush=True)

if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("phase",choices=("prepare","extract","evaluate"))
    phase=p.parse_args().phase
    if phase=="prepare": prepare()
    else:
        raw=(OUT/"stimuli.jsonl").read_bytes()
        lock=json.loads((OUT/"config.json").read_text(encoding="utf-8"))
        assert hashlib.sha256(raw).hexdigest()==lock["stimuli_sha256"]
        assert hashlib.sha256(Path(__file__).read_bytes()).hexdigest()==lock["source_sha256"]
        rows=[json.loads(line) for line in raw.splitlines()]
        (extract if phase=="extract" else evaluate)(rows)
