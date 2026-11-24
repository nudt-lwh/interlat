"""Check each paired instruction and original order arithmetic independently of the model."""
from collections import Counter,defaultdict
from prepare import *

def amounts(r):
    return [r["qty"]*(p[2] if p[1]=="unit price" else r["price"])+(p[2] if p[1]=="delivery fee" else r["fee"]) for p in r["proposals"]]

def summarize(rows):
    bypair=defaultdict(list)
    for r in rows:
        bypair[r["group"],r["pair"]].append(r)
    result={"instructions":len(rows),"pairs":len(bypair)}
    for label in ("pred","zero_pred","grounded_pred"):
        result[label]=dict(single_correct=sum(r[label]==r["gold"] for r in rows),
             pair_both_correct=sum(all(r[label]==r["gold"] for r in rr) for rr in bypair.values()))
    result["newly_correct_vs_zero"]=sum(r["zero_pred"]!=r["gold"]==r["pred"] for r in rows)
    result["newly_wrong_vs_zero"]=sum(r["pred"]!=r["gold"]==r["zero_pred"] for r in rows)
    return result

def main():
    raw=load_data(test)
    rows=load_data(folder/"rows.jsonl")
    assert len(raw)==len(rows)==600
    paired=defaultdict(list)
    for i,(r,orig) in enumerate(zip(rows,raw),1):
        assert r["source_line"]==i and r["text"]==orig["text"]
        assert all(r[k]==orig[k] for k in ("group","variant","pair"))
        assert r["zero_pred"]==orig["direct_choice"] and r["grounded_pred"]==orig["pred"]
        assert r["gold"]==orig["answer"] and r["pred"]==int(r["logit_B"]>r["logit_A"])
        totals=amounts(orig)
        assert totals[0]!=totals[1]
        assert r["pred_total"]==totals[r["pred"]] and r["gold_total"]==orig["total"]==totals[r["gold"]]
        paired[r["group"],r["pair"]].append((r,orig))
    assert len(paired)==300
    with (folder/"pairs.jsonl").open("w") as f:
        for (g,k),items in paired.items():
            assert len(items)==2
            (a,oa),(b,ob)=items
            assert [a["gold"],b["gold"]] in ([0,1],[1,0])
            assert all(oa[k]==ob[k] for k in ("qty","price","fee","proposals","days","first_offer")) and a["gold_total"]!=b["gold_total"]
            f.write(json.dumps(dict(group=g,pair=k,source_lines=[a["source_line"],b["source_line"]],gold=[a["gold"],b["gold"]],
                       few=[a["pred"],b["pred"]],few_both=a["pred"]==a["gold"] and b["pred"]==b["gold"],
                       zero_both=a["zero_pred"]==a["gold"] and b["zero_pred"]==b["gold"]))+"\n")
    overall=summarize(rows)
    assert overall["zero_pred"]=={"single_correct":396,"pair_both_correct":99}
    demonstrations=json.loads((folder/"demos.json").read_text())
    assert demonstrations==select()
    dev_rows=load_data(dev)
    for dm in demonstrations:
        o=dev_rows[dm["dev_line"]-1]
        assert o["text"]==dm["text"] and "AB"[o["answer"]]==dm["answer"]
    assert len({r["text"] for r in rows} & {d["text"] for d in demonstrations})==0
    result=dict(overall=overall,groups={name:summarize([r for r in rows if r["group"]==g]) for g,name in enumerate(("supplier_name","offer_order","delivery_duration"))},
                few_shot_correct_amount=sum(r["pred_total"]==r["gold_total"] for r in rows),prompt_tokens=[min(r["tokens"] for r in rows),max(r["tokens"] for r in rows)])
    save("summary.json",result)
    print(json.dumps(result,indent=2))

if __name__=="__main__":
    main()
