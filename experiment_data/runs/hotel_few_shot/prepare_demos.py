"""Build four fixed few-shot demonstrations exclusively from MultiWOZ's development split."""
import argparse
import hashlib
import json
import random
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent/"hotel_replay"))
from run import NUMBER, num, oracle, dump, prompt

SOURCE = "994fec69745c3dfd469835e90bf5cba2d1c4af47"
SEED = 853991

def candidates(path):
    raw = path.read_bytes()
    assert hashlib.sha1(b"blob "+str(len(raw)).encode()+b"\0"+raw).hexdigest() == SOURCE
    pool = {"guests": [], "nights": []}
    for conversation in json.loads(raw):
        previous, context = {}, ""
        for t in conversation["turns"]:
            if t["speaker"] == "SYSTEM":
                context = t["utterance"]
                continue
            for f in t["frames"]:
                if f["service"] != "hotel":
                    continue
                state = f["state"]["slot_values"]
                names = {"hotel-bookpeople": "guests", "hotel-bookstay": "nights"}
                changed = [k for k in names if k in state and state[k] != previous.get(k)]
                previous = state
                ns = re.findall(NUMBER,t["utterance"].lower())
                if len(changed) != 1 or len(ns) != 1:
                    continue
                key = changed[0]
                if str(num(ns[0])) != state[key][0]:
                    continue
                # Demos have explicit units for reliable development labels.
                cue = r"\b(people|person|persons|guests)\b" if names[key]=="guests" else r"\bnights?\b"
                if not re.search(cue,t["utterance"],re.I):
                    continue
                pool[names[key]].append(dict(id=conversation["dialogue_id"]+"/"+t["turn_id"],
                    context=context,text=t["utterance"],field=names[key],value=num(ns[0])))
    return pool

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("dev",type=Path)
    path=parser.parse_args().dev
    pool=candidates(path)
    rng=random.Random(SEED)
    for options in pool.values():
        rng.shuffle(options)
    demos=[]
    used=set()
    for fields in [["guests"],["nights"],["guests","nights"],["nights","guests"]]:
        initial=[rng.randint(2,4),rng.randint(2,4),rng.randint(9,32),rng.randint(1,6),rng.randint(1,7),rng.randint(1,9)]
        desc=(f"Guests: {initial[0]}. Nights: {initial[1]}. Room rate per guest per night: {initial[2]}. "
              f"One-time linen fee per guest: {initial[3]}. Service fee per night: {initial[4]}. "
              f"One-time booking fee: {initial[5]}. All charges in dollars.")
        state=initial.copy()
        turns=[]
        for field in fields:
            index=int(field=="nights")
            item=next(u for u in pool[field] if u["id"].split("/")[0] not in used and u["value"]!=state[index])
            used.add(item["id"].split("/")[0])
            state[index]=item["value"]
            turns.append(item)
        demos.append(dict(initial=desc,turns=turns,gold=oracle(state),messages=[
            dict(role="user",content=prompt(desc,turns)),dict(role="assistant",content=str(oracle(state)))]))
    dump(HERE/"demos.json",demos)

if __name__=="__main__":
    main()
