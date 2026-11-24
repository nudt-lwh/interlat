"""Original public MultiWOZ turns, assembled into counterfactual hotel bills."""
import argparse
import hashlib
import json
import random
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from statistics import median
from time import perf_counter_ns

HERE = Path(__file__).resolve().parent
BASELINE = HERE.parent / "2026-11-19_方法迭代"
sys.path.insert(0, str(BASELINE))
from iterate import Candidate, truth

SHARDS = ["fd420005cc9dc259892f541b001d9d48608bd38d", "8ce97bc17acc3d1105d92d8d936520b8c93892ba"]
WORDS = "zero one two three four five six seven eight nine ten".split()
NUMBER = r"\b(?:\d+|" + "|".join(WORDS) + r")\b"
CUES = {"guests": r"\b(?:people|persons?|guests?|travell?ers?|adults?|party|occupants?|friends?|family|myself)\b",
        "nights": r"\b(?:nights?|days?|duration|long|overnight)\b"}
# guests, nights, room rate per person-night, linen per person, per-night service, fixed fee
GRAPH = [(2, 0, 1), (2, 6, 2), (2, 0, 3), (2, 1, 4), (0, 7, 8), (0, 10, 9), (0, 11, 5)]
SEED = 194861
FIELDS = ["guests", "nights"]

def num(token):
    return int(token) if token.isdigit() else WORDS.index(token)

def read_json(p):
    return json.loads(p.read_text(encoding="utf-8"))

def dump(p, x):
    p.write_text(json.dumps(x, indent=2, ensure_ascii=False), encoding="utf-8")

def identify(context, utterance):
    tokens = list(re.finditer(NUMBER, utterance.lower()))
    if len(tokens) != 1:
        return None, None
    token = tokens[0]
    for phrase in [utterance.lower(), context.lower()]:
        hits = [key for key, pattern in CUES.items() if re.search(pattern, phrase)]
        if len(hits) == 1:
            return hits[0], num(token.group())
        if len(hits) == 2:
            return None, None
    return None, None

def corpus(folder):
    entries = []
    for i, sha in enumerate(SHARDS, 1):
        raw = (folder / f"test_{i:03}.json").read_bytes()
        assert hashlib.sha1(b"blob " + str(len(raw)).encode() + b"\0" + raw).hexdigest() == sha
        for dialogue in json.loads(raw):
            previous, system = {}, ""
            for turn in dialogue["turns"]:
                if turn["speaker"] == "SYSTEM":
                    system = turn["utterance"]
                    continue
                for frame in turn["frames"]:
                    if frame["service"] != "hotel":
                        continue
                    state = frame["state"]["slot_values"]
                    slots = {"hotel-bookpeople":"guests", "hotel-bookstay":"nights"}
                    changed = [s for s in slots if s in state and state.get(s) != previous.get(s)]
                    previous = state
                    if len(changed) != 1:
                        continue
                    key = changed[0]
                    val = state[key][0]
                    numbers = re.findall(NUMBER, turn["utterance"].lower())
                    if len(numbers) == 1 and val.isdigit() and num(numbers[0]) == int(val):
                        entries.append(dict(id=dialogue["dialogue_id"] + "/" + turn["turn_id"], shard=i,
                                            context=system, text=turn["utterance"], field=slots[key], value=int(val)))
    return entries

def prepare(folder):
    pool = corpus(folder)
    # Implicit references to a different domain cannot be interpreted as standalone amendments.
    excluded = set(read_json(HERE / "exclusions.json"))
    pool = [u for u in pool if u["id"] not in excluded]
    rng, usage, cases = random.Random(SEED), Counter(), []
    def pick(predicate, seen):
        options = [e for e in pool if predicate(e) and e["id"].split("/")[0] not in seen]
        lowest = min(usage[e["id"]] for e in options)
        entry = rng.choice([e for e in options if usage[e["id"]] == lowest])
        usage[entry["id"]] += 1
        seen.add(entry["id"].split("/")[0])
        return entry
    for pair in range(24):
        length = 3 + pair % 3
        target = FIELDS[pair % 2]
        initial = [rng.randint(2,4), rng.randint(2,3), rng.randint(9,32), rng.randint(1,6), rng.randint(1,7), rng.randint(1,9)]
        # The fork points upwards/downwards from exactly the same pre-fork state.
        seen, prefix = set(), []
        state = dict(zip(FIELDS, initial))
        for k in range(length-1):
            field = FIELDS[(pair+k+1) % 2]
            viable = sorted({e["value"] for e in pool if e["field"] == target})
            e = pick(lambda u: u["field"] == field and u["value"] != state[field] and
                     (field != target or viable[0] < u["value"] < viable[-1]), seen)
            prefix.append(e)
            state[field] = e["value"]
        for direction in [-1, 1]:
            final = pick(lambda u: u["field"] == target and (u["value"] - state[target])*direction > 0, seen)
            desc = (f"Guests: {initial[0]}. Nights: {initial[1]}. Room rate per guest per night: {initial[2]}. "
                    f"One-time linen fee per guest: {initial[3]}. Service fee per night: {initial[4]}. "
                    f"One-time booking fee: {initial[5]}. All charges in dollars.")
            cases.append(dict(pair=pair, direction=direction, initial=desc, turns=prefix+[final]))
    dump(HERE/"cases.json", cases)
    return cases

def initial_values(desc):
    return list(map(int, re.findall(r"\d+", desc)))

def oracle(values):
    g,n,r,l,s,b = values
    return g*n*r + g*l + n*s + b

def rebuild(values, encode, execute):
    state = [encode(x) for x in values]
    for op, l, r in GRAPH:
        state.append(execute(op, state[l], state[r]))
    return state

def cpu(cases):
    model=Candidate(42, True, True)
    rows = []
    for case in cases:
        for method in ("learned_local","symbolic_local","learned_rebuild"):
            encode = (lambda x: model.encode("s",x)) if method != "symbolic_local" else int
            execute = model.execute if method != "symbolic_local" else truth
            start = perf_counter_ns()
            values = initial_values(case["initial"])
            state = rebuild(values, encode, execute)
            init_ns = perf_counter_ns()-start
            gold = values.copy()
            for step,u in enumerate(case["turns"],1):
                start = perf_counter_ns()
                field,value = identify(u["context"],u["text"])
                if field is not None:
                    index = FIELDS.index(field)
                    values[index] = value
                    if method == "learned_rebuild":
                        state = rebuild(values, encode, execute)
                    else:
                        state[index] = encode(value)
                        dirty = {index}
                        for j,(op,l,r) in enumerate(GRAPH, len(values)):
                            if l in dirty or r in dirty:
                                state[j] = execute(op,state[l],state[r])
                                dirty.add(j)
                prediction = round(state[-1])
                ns = perf_counter_ns()-start
                gold[FIELDS.index(u["field"])] = u["value"]
                rows.append(dict(pair=case["pair"],direction=case["direction"],step=step,length=len(case["turns"]),
                                 method=method,id=u["id"],field=u["field"],value=u["value"],identified=field,
                                 parsed=value,field_correct=field==u["field"],prediction=prediction,truth=oracle(gold),
                                 correct=prediction==oracle(gold), ns=ns, init_ns=init_ns))
    return rows

def prompt(desc, history):
    edits = "\n".join(f"Clerk context: {u['context']}\nCustomer: {u['text']}" for u in history)
    return ("Reprice this hotel booking. Apply each customer's new guest count or stay length in order. "
            "Each exchange is a separate amendment. Ignore unrelated requests. Use the clerk's context "
            "only to interpret that customer's words. Keep other bill terms fixed.\n"
            + desc + "\n" + edits + "\nWhat is the updated total cost in dollars? Output only the whole number.")

def direct(cases, device):
    import torch, transformers
    from transformers import AutoTokenizer, AutoModelForCausalLM
    base=HERE.parents[1]/"artifacts/Qwen3-1.7B"
    tokenizer=AutoTokenizer.from_pretrained(base, local_files_only=True)
    lm=AutoModelForCausalLM.from_pretrained(base, dtype=torch.bfloat16,local_files_only=True).to(device).eval()
    def answer(desc,history):
        chat = tokenizer.apply_chat_template([{"role":"user","content":prompt(desc,history)}],tokenize=False,
                                             add_generation_prompt=True,enable_thinking=False)
        inputs=tokenizer(chat,return_tensors="pt").to(device)
        with torch.inference_mode():
            gen=lm.generate(**inputs,do_sample=False,max_new_tokens=16,pad_token_id=tokenizer.eos_token_id)
        output=tokenizer.decode(gen[0,inputs.input_ids.shape[1]:],skip_special_tokens=True).strip()
        m=re.fullmatch(r"\$?\s*([0-9][0-9,]*)(?:\.0+)?\s*(?:dollars?|USD)?[.!]?",output,re.I)
        return (int(m[1].replace(",","")) if m else None),output
    answer("Guests: 2. Nights: 3. Rate: 1 dollar per guest per night.",[])
    rows=[]
    for case in cases:
        gold=initial_values(case["initial"])
        for step,u in enumerate(case["turns"],1):
            if device=="cuda":torch.cuda.synchronize()
            start=perf_counter_ns()
            pred,raw=answer(case["initial"],case["turns"][:step])
            if device=="cuda":torch.cuda.synchronize()
            elapsed=perf_counter_ns()-start
            gold[FIELDS.index(u["field"])]=u["value"]
            rows.append(dict(pair=case["pair"], direction=case["direction"],step=step,length=len(case["turns"]),
                             id=u["id"],method="qwen_direct",prediction=pred,raw=raw,truth=oracle(gold),
                             correct=pred==oracle(gold),ns=elapsed))
        print("finished",case["pair"],case["direction"],flush=True)
    dump(HERE/"direct_env.json",dict(torch=torch.__version__,transformers=transformers.__version__,device=device,
                                     gpu=torch.cuda.get_device_name() if device=="cuda" else None))
    return rows

def summarize(rows):
    result={}
    for method in sorted({r["method"] for r in rows}):
        sub=[r for r in rows if r["method"]==method]
        seq=defaultdict(list)
        for r in sub:seq[r["pair"],r["direction"]].append(r)
        final=[s[-1] for s in seq.values()]
        result[method]=dict(steps=len(sub),correct_steps=sum(r["correct"] for r in sub),
            sequences=len(seq),all_steps_correct=sum(all(r["correct"] for r in s) for s in seq.values()),
            final_correct=sum(r["correct"] for r in final),
            pairs=len(final)//2,both_finals_correct=sum(seq[p,-1][-1]["correct"] and seq[p,1][-1]["correct"] for p,d in seq if d==-1),
            median_step_ms=median(r["ns"] for r in sub)/1e6,mean_step_ms=sum(r["ns"] for r in sub)/len(sub)/1e6,
            distinct_source_turns=len({r["id"] for r in sub}))
        if "field_correct" in sub[0]:
            result[method].update(field_correct=sum(r["field_correct"] for r in sub), median_init_ms=median(r["init_ns"] for r in sub)/1e6,
                unique_turns_identified=len({r["id"] for r in sub if r["field_correct"]}))
    dump(HERE/"summary.json",result)

if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("phase",choices=["prepare","cpu","direct","summary"])
    parser.add_argument("--folder",type=Path)
    parser.add_argument("--device",default="cuda")
    args=parser.parse_args()
    if args.phase=="prepare":
        prepare(args.folder)
    elif args.phase in ("cpu","direct"):
        cases=read_json(HERE/"cases.json")
        config=read_json(HERE/"config.json")
        for f,d in config["locked_sha256"].items():
            assert hashlib.sha256((HERE/f).read_bytes()).hexdigest()==d,f
        rows=cpu(cases) if args.phase=="cpu" else direct(cases,args.device)
        (HERE/f"{args.phase}_rows.jsonl").write_text("".join(json.dumps(r)+"\n" for r in rows),encoding="utf8")
    else:
        summarize([json.loads(l) for f in HERE.glob("*_rows.jsonl") for l in f.read_text(encoding="utf8").splitlines()])
