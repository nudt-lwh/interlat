"""Aggregate paired output and derive task-level bootstrap intervals."""
import collections
import json
import math
import random
import statistics
from pathlib import Path

root=Path(__file__).resolve().parent


def load(name):
    return [json.loads(s) for s in (root/name).read_text(encoding="utf8").splitlines()]


def mean(values):
    return statistics.mean(values)


def ci(values, stat):
    rng=random.Random(751)
    draws=sorted(stat(rng.choices(values,k=len(values))) for _ in range(1500))
    return [draws[37],draws[-38]]


def fork():
    rows=load("fork_rows.jsonl")
    assert len({(r['width'],r['task'],r['private']) for r in rows})==len(rows)==1200
    groups=[]
    for w in (2,4,8):
        for private in (True,False):
            group=[r for r in rows if (r["width"],r["private"])==(w,private)]
            assert len(group)==200
            def speed(rs):
                return sum(r["rebuild_ms"] for r in rs)/sum(r["local_ms"] for r in rs)
            groups.append(dict(width=w,private=private,n=len(group),
               local=sum(r["local"]==r["truth"] for r in group),
               rebuild=sum(r["rebuild"]==r["truth"] for r in group),
               agree=sum(r["local"]==r["rebuild"] for r in group),
               both_wrong=sum(r["local"]!=r["truth"] and r["rebuild"]!=r["truth"] for r in group),
               retained=sum(r["preserved"] for r in group),
               expected_match=sum(r["expected"]==r["executed"] for r in group),
               executed=mean([len(r["executed"]) for r in group]),total_derived=2*w-1,
               correct_and_skipped=sum(r["local"]==r["truth"] and len(r["executed"])<2*w-1 for r in group),
               speed=speed(group),speed_ci=ci(group,speed),
               medians_ms={k:statistics.median(r[k] for r in group) for k in
                           ("local_ms","rebuild_ms","local_update_ms","rebuild_update_ms")}))
    return groups


def belief():
    rows=load("belief_rows.jsonl")
    assert len({r["id"] for r in rows})==len(rows)==641
    runs=Path("experiment_data/runs")
    methods={}
    def metrics(rs,key):
        bu=[r for r in rs if r["gt0"]!=r["gt1"]]
        bm=[r for r in rs if r["gt0"]==r["gt1"]]
        def count(group):
            return sum(r[key]==r["gt1"] for r in group)
        def joint(group):
            return sum(r[key]==r["gt1"] and r["initial"]==r["gt0"] for r in group)
        return dict(n=len(rs),init=sum(r["initial"]==r["gt0"] for r in rs),
                    bu=count(bu),bu_n=len(bu),bm=count(bm),bm_n=len(bm),
                    breu=(count(bu)/len(bu)+count(bm)/len(bm))/2,
                    correct_update=joint(bu),correct_maintain=joint(bm),
                    initially_correct_bu=sum(r["initial"]==r["gt0"] for r in bu),
                    initially_correct_bm=sum(r["initial"]==r["gt0"] for r in bm),
                    transitions=dict(collections.Counter(f"{r['initial']}>{r[key]}" for r in rs)))
    for key in ("edited","initial","rebuild"):
        methods[key]=metrics(rows,key)
    for name in ("belief_graph_v4","llm_qwen_full","llm_llama_full"):
        raw=[json.loads(s) for s in (runs/name/"rows.jsonl").read_text().splitlines()]
        converted=[{k:"abc".index(r[col]) for k,col in [("initial","a0"),("edited","a1"),("gt0","gt0"),("gt1","gt1")]} for r in raw]
        assert {r["id"] for r in raw}=={r["id"] for r in rows}
        methods[name]=metrics(converted,"edited")
    methods["local_rebuild_agree"]=sum(r["edited"]==r["rebuild"] for r in rows)
    rng=random.Random(43)
    groups=[[int(r['edited']==r['gt1'])-int(r['initial']==r['gt1']) for r in rows if (r['gt0']!=r['gt1'])==change] for change in (True,False)]
    draws=sorted(sum(mean(rng.choices(g,k=len(g))) for g in groups)/2 for _ in range(2000))
    methods["breu_gain_over_copy_ci"]=[draws[50],draws[-51]]
    return methods


if __name__=="__main__":
    for name,fn in (("fork",fork),("belief",belief)):
        if (root/(name+"_rows.jsonl")).exists():
            result=fn()
            (root/(name+"_summary.json")).write_text(json.dumps(result,indent=2))
            print(name,result)
