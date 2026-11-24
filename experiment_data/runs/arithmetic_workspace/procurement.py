"""Paired synthetic procurement notices. The same two quotes receive opposite decisions."""
import argparse
import hashlib
import json
import random
import re
import sys
from pathlib import Path

import torch
from transformers import AutoTokenizer,AutoModelForCausalLM
from iterate import Candidate,out

def grounding(r):
    # Only the written decision and quote metadata are visible to the resolver.
    tokens=set(re.findall("[a-z]+",r["decision"].lower()))
    faster={"shorter","shortest","sooner","soonest","earlier","earliest","faster","fastest","quickest","fewer"}
    slower={"longer","longest","later","latest","slower","slowest","more"}
    fast=int(r["days"][1]<r["days"][0])
    if tokens & {"time","days","schedule","deliver","delivery","shipment"}:
        if bool(tokens&faster)!=bool(tokens&slower):return (fast if tokens&faster else 1-fast),"duration"
    rejected=bool(tokens&{"decline","declined","reject","rejected","withdrawn","discard","skip","denied","cancelled","canceled","lost","ignore","eliminated","drop","disqualified","exclude","excluded","refused","down"})
    if ("first" in tokens)!=("second" in tokens):
        idx=r["first_offer"] if "first" in tokens else 1-r["first_offer"]
        return (1-idx if rejected else idx),"order"
    ids=[i for i,p in enumerate(r["proposals"]) if p[0].lower() in tokens]
    if len(ids)==1:return (1-ids[0] if rejected else ids[0]),"name"
    return None,"fallback"


def cases(phase):
    dev=phase=="dev"
    rng=random.Random(1009 if dev else 693821)
    names=["Alder","Beacon","Cedar","Dune","Ember","Falcon"] if dev else ["Kestrel","Linden","Meridian","Northgate","Orchid","Pioneer"]
    texts0=[
        ("Use the revision offered by {yes}.","Decline the offer from {no}. Use the other revision."),
        ("Adopt the {order} offer in the list above.","Skip the {opp} offer and process the remaining one."),
        ("Select the supplier with the {quality} lead time.","Select the {quality} delivery schedule."),
    ]
    texts1=[
        ("The purchasing manager signed off on {yes}'s amendment.","{no}'s amendment has been turned down. Its competitor got the order."),
        ("Approval went to the {order} of the two entries above.","The {opp} entry above was withdrawn. The remaining entry is final."),
        ("The award goes to whichever vendor can deliver {when}.","Delivery timing determines the award. Choose the {quality} lead time."),
    ]
    rows=[]
    for group in range(3):
        for i in range(18 if dev else 100):
            a,b=rng.sample(names,2);quantity=rng.randint(3,12)
            price,fee=rng.sample(range(12,75),2)
            while True:
                pn,fn=rng.sample(range(12,90),2)
                totals=[quantity*pn+fee,quantity*price+fn]
                if pn!=price and fn!=fee and totals[0]!=totals[1]:break
            proposals=[(a,"unit price",pn,0),(b,"delivery fee",fn,1)]
            rng.shuffle(proposals)
            days=rng.sample(range(2,16),2)
            lines=[f"{n} offered to revise the {field} to {value} dollars (delivery in {days[k]} days)."
                   for k,(n,field,value,_) in enumerate(proposals)]
            variant=i%2
            for target in (0,1):
                opts=(texts0 if dev else texts1)[group]
                fast=days[target]<days[1-target]
                decision=opts[variant].format(yes=proposals[target][0],no=proposals[1-target][0],
                          order=("first","second")[target],opp=("second","first")[target],
                          quality="shorter" if fast else "longer",when="sooner" if fast else "later")
                option_order=[0,1] if i%4 in (0,3) else [1,0]
                choices=[proposals[z][0] for z in option_order]
                prompt=(f"Current order: {quantity} units at {price} dollars each, plus a delivery fee of {fee} dollars.\n"
                        +"\n".join(lines)+"\nFinal purchasing note: "+decision+
                        f"\nWhich supplier's revision should be applied?\nA. {choices[0]}\nB. {choices[1]}\nAnswer with just A or B.")
                rows.append(dict(group=group,pair=i,variant=variant,text=prompt,answer=option_order.index(target),
                    decision=decision,days=[days[z] for z in option_order],first_offer=option_order.index(0),
                    qty=quantity,price=price,fee=fee,proposals=[proposals[z] for z in option_order],
                    total=totals[proposals[target][3]],last_offer=option_order.index(1)))
    return rows


def main():
    phase=sys.argv[1];lock=out/"procurement_frozen.json"
    if phase=="formal":
        assert json.loads(lock.read_text())["sha256"]==hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    rows=cases(phase)
    base=out.parents[1]/"interlat-code/experiment_data/artifacts/Qwen3-1.7B"
    tok=AutoTokenizer.from_pretrained(base,padding_side="left")
    lm=AutoModelForCausalLM.from_pretrained(base,dtype=torch.bfloat16).cuda().eval()
    import os
    protocol=os.environ.get("PROC_METHOD","direct")
    letters=[tok.encode(s,add_special_tokens=False) for s in ("ABCDEF" if protocol=="policy" else "AB")]
    assert all(len(l)==1 for l in letters)
    cols=[l[0] for l in letters]
    typed=Candidate(42,True,True)
    with torch.inference_mode():
        for i in range(0,len(rows),8):
            sub=rows[i:i+8]
            def prompt(r):
                if protocol in ("direct","grounded"): return r["text"]
                a,b=[p[0] for p in r["proposals"]]
                if protocol=="facts":
                    first=r["first_offer"];fast=int(r["days"][1]<r["days"][0])
                    vendors=[a,b]
                    return (f"First entry: {vendors[first]}. Second entry: {vendors[1-first]}.\n"
                            f"Shorter lead time (earlier delivery): {vendors[fast]}. Longer lead time (later delivery): {vendors[1-fast]}.\n"
                            f"Purchasing manager: {r['decision']}\n"
                            "Withdrawn or declined means the other entry wins. Only identify the winning vendor.\n"
                            f"A. {a}\nB. {b}\nWinning vendor, A or B?")
                return (f"Interpret the purchasing manager's final decision. Two suppliers are {a} and {b}.\n"
                        f"Decision: {r['decision']}\nChoose the policy that implements the decision.\n"
                        f"A: use {a}\nB: use {b}\nC: use the first entry\nD: use the second entry\n"
                        "E: use the supplier with fewer delivery days\nF: use the supplier with more delivery days\n"
                        "Check whether the note rejects an entry or approves it. Answer with one letter.")
            txt=[tok.apply_chat_template([dict(role="user",content=prompt(r))],tokenize=False,
                                add_generation_prompt=True,enable_thinking=False) for r in sub]
            t=tok(txt,padding=True,return_tensors="pt").to("cuda")
            scores=lm(**t,logits_to_keep=1,use_cache=False).logits[:,-1,cols]
            for r,idx in zip(sub,scores.argmax(-1).tolist()):
                r["direct_choice"]=idx
                if protocol=="grounded":
                    grounded,source=grounding(r)
                    r["source"]=source
                    if grounded is not None:idx=grounded
                if protocol=="policy":
                    fast=int(r["days"][1]<r["days"][0])
                    r["policy"]=idx
                    idx=[0,1,r["first_offer"],1-r["first_offer"],fast,1-fast][idx]
                _,field,value,_=r["proposals"][idx]
                qty,price,fee=[typed.encode("s",r[k]) for k in ("qty","price","fee")]
                subtotal=typed.execute(2,qty,price)
                if field=="unit price":
                    subtotal=typed.execute(2,qty,typed.encode("s",value))
                else: fee=typed.encode("s",value)
                r["pred"]=idx;r["pred_total"]=round(typed.execute(0,subtotal,fee))
    report={}
    for g in range(3):
        sub=[r for r in rows if r["group"]==g]
        report[g]=dict(n=len(sub),choice=sum(r["pred"]==r["answer"] for r in sub),
            correct_total=sum(r["pred_total"]==r["total"] for r in sub),
            last_offer=sum(r["last_offer"]==r["answer"] for r in sub),
            both=sum(all(r["pred_total"]==r["total"] for r in sub if r["pair"]==j)
                     for j in {r["pair"] for r in sub}))
    print(report,flush=True)
    (out/f"procurement_{phase}_{protocol}.jsonl").write_text("".join(json.dumps(r)+"\n" for r in rows))
    (out/f"procurement_{phase}_{protocol}_summary.json").write_text(json.dumps(report,indent=2))

if __name__=="__main__":main()
