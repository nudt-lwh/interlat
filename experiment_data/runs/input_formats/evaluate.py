"""Use saved fit 42; paired original edits with frozen text contract, explicit adapters and token probes."""
import itertools
import re
from collections import Counter
from functools import lru_cache
from encoder import *

WORDS = "zero one two three four five six seven eight nine".split()
NEW = ["alpha"]+[f"gamma_{i:02d}" for i in range(12)]
NAMES = ["s"] + [f"x{i}" for i in range(12)]
CASES = [("original","contract"),("rename","contract"),("sentence","contract"),("words","contract"),
         ("sentence","adapter"),("words","adapter"),("rename","probe"),("sentence","probe"),("words","probe")]

def render(style, name, value):
    if style == "rename":
        name = NEW[NAMES.index(name)]
    if style == "words":
        return name + " = " + " ".join(WORDS[int(c)] for c in value)
    if style == "sentence":
        return f"the value of {name} is {value}"
    return f"{name} = {value}"

def main():
    torch.set_num_threads(4)
    model = Candidate(42,True,True)
    strings = sorted({render(style,name,str(d)) for style in ("original","rename","sentence","words")
                      for name in (NAMES if style=="rename" else ["s"]) for d in range(10)})
    enc = FrozenEncoder("cuda")
    features = np.concatenate([enc(strings[i:i+12]) for i in range(0,len(strings),12)])
    np.savez(here/"features.npz",strings=strings,z=features)
    digits = dict(zip(strings,(features-model.mean)@model.weights+model.bias))
    # Match the original extract.py first batch: names outermost, digits innermost.
    calibration_batch = [f"s = {i}" for i in range(10)]+["x0 = 0","x0 = 1"]
    fixed = enc(calibration_batch)[:10]
    archived = np.load(method/"inputs.npz")["z"][0]
    batch_delta = float(np.max(np.abs(fixed-archived)))
    fixed_digits = dict(zip(calibration_batch[:10],(fixed-model.mean)@model.weights+model.bias))
    mixed_digits = digits.copy()
    primary = [(s,r) for s,r in CASES]
    diagnostic = [(s,r+"_mixed_batch") for s,r in CASES if r != "probe"]
    source = formal_rows()
    results = {}
    with (here/"rows.jsonl").open("w") as out:
        for style,condition in primary + diagnostic:
            route = condition.split("_mixed_batch")[0]
            digits = mixed_digits if condition.endswith("_mixed_batch") or route == "probe" else fixed_digits
            @lru_cache(None)
            def decode(raw):
                if route == "contract":
                    m = re.fullmatch(r"(\w+) = ([0-9]+)",raw)
                    if not m: return None
                    return number(model,[digits[f"s = {c}"] for c in m[2]])
                if style == "sentence":
                    m = re.fullmatch(r"the value of (\w+) is ([0-9]+)",raw)
                    tokens=list(m[2])
                    keys = [f"the value of s is {t}" if route=="probe" else f"s = {t}" for t in tokens]
                elif style == "words":
                    tokens=raw.split(" = ")[1].split()
                    keys=[f"s = {t}" if route=="probe" else f"s = {WORDS.index(t)}" for t in tokens]
                else:
                    name,value=raw.split(" = ")
                    keys=[f"{name} = {t}" for t in value]
                return number(model,[digits[k] for k in keys])
            counts=Counter()
            for idx,row in enumerate(source):
                target=row["target"]
                raw=[render(style,n,str(v)) for n,v in zip(NAMES,row["inputs"])]
                changed=render(style,NAMES[target],str(row["after"][target]))
                vals=[decode(t) for t in raw]
                new=decode(changed)
                parsed=new is not None and all(v is not None for v in vals)
                pred=rebuild=preserved=agree=None
                if parsed:
                    graph=row["graph"]
                    state=build(vals,graph,model.execute)
                    local,dirty=edit(model,state,target,new,graph,len(vals))
                    vals[target]=new
                    pred=round(local[-1])
                    rebuild=round(build(vals,graph,model.execute)[-1])
                    preserved=all(local[j]==v for j,v in enumerate(state) if j not in dirty)
                    agree=pred==rebuild
                truth=row["truth"]
                counts.update(n=1,covered=int(parsed),correct=int(pred==truth),agree=int(agree is True),preserved=int(preserved is True))
                out.write(json.dumps(dict(source_line=idx+1,style=style,route=condition,edited_text=changed,
                            target=target,truth=truth,pred=pred,rebuild=rebuild,covered=parsed,correct=pred==truth,
                            preserved=preserved,agreement=agree))+"\n")
            results[f"{style}/{condition}"]=dict(counts)
    for stats in results.values():
        stats["exact_pct"]=100*stats["correct"]/stats["n"]
        stats["delta_pp"]=stats["exact_pct"]-100*results["original/contract"]["correct"]/results["original/contract"]["n"]
    (here/"summary.json").write_text(json.dumps(results,indent=2))
    (here/"config.json").write_text(json.dumps(dict(fit_seed=42,model="Qwen3-1.7B with archived encoder LoRA and projection",device=torch.cuda.get_device_name(),
        source="arithmetic_workspace/formal_rows_42.jsonl; source_line is 1-based",cases=primary+diagnostic,original_batch_max_feature_difference=batch_delta,
        original_names=NAMES,heldout_names=NEW,words=WORDS,refit=False,
        grammar="Original API accepts structured name and digit string; contract measures a literal assignment interface restricted to name = ASCII digits. Failure means input outside that interface.",
        probe="Oracle slot and digit segmentation, no refit. Rename bypasses canonical-name replacement. Sentence and word probes feed changed per-digit strings with canonical s to the encoder.",
        adapter="Sentence adapter extracts the supplied numeric slot. Word adapter uses explicit English-to-Arabic digit dictionary. All input/edit strings are rendered then parsed. Graph/target given.",
        cache="Frozen contextual features memoized per distinct literal; agreement/preservation undefined if no state. Null is not success."),indent=2))
    print(results,flush=True)

if __name__=="__main__": main()

