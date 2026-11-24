"""Replay fixed Qwen test items with Llama digit vectors and refit the original calibrator."""
import hashlib
import json
from pathlib import Path
import sys

here = Path(__file__).resolve().parent
original = here.parent / "arithmetic_workspace"
sys.path.insert(0, str(original))
import iterate

def main():
    iterate.out = here
    seed = 42
    model = iterate.Candidate(seed, True, True)
    data = original / "formal_rows_42.jsonl"
    config = dict(encoder="Llama-3.2-1B-Instruct", encoder_checkpoint="llama_s2_mul/ckpt_final.pt",
                  fit_seed=seed, pool="masked mean; arithmetic mean across two 256-dimensional projection slots",
                  qwen_baseline_correct=10800, test_file=data.name,
                  test_sha256=hashlib.sha256(data.read_bytes()).hexdigest(),
                  features_sha256=hashlib.sha256((here/"inputs.npz").read_bytes()).hexdigest())
    (here/"config.json").write_text(json.dumps(config,indent=2))
    correct = agree = preserved = count = 0
    with (here/"rows.jsonl").open("w") as output:
        for line in data.read_text().splitlines():
            case=json.loads(line)
            before,after,nodes,target = [case[k] for k in ("inputs","after","graph","target")]
            def encode(values):
                return [model.encode("s", v) for v in values]
            state = iterate.build(encode(before),nodes,model.execute)
            cache = state.copy()
            cache[target] = model.encode("s",after[target])
            dirty={target}
            for index,(op,l,r) in enumerate(nodes,len(before)):
                if l in dirty or r in dirty:
                    cache[index]=model.execute(op,cache[l],cache[r])
                    dirty.add(index)
            pred=round(cache[-1])
            rebuilt=round(iterate.build(encode(after),nodes,model.execute)[-1])
            kept=all(v==state[i] for i,v in enumerate(cache) if i not in dirty)
            result={k:case.get(k) for k in ("width","mixed","task","range","target","truth")}
            result.update(row=count, prediction=pred,rebuild=rebuilt,preserved=kept,qwen_prediction=case["pred"])
            output.write(json.dumps(result)+"\n")
            count+=1
            correct+=int(pred==case["truth"])
            agree+=int(pred==rebuilt)
            preserved+=int(kept)
    assert count == 10800
    summary=dict(n=count,correct=correct,agreement=agree,preserved=preserved,
                 delta_pp=100*(correct-config["qwen_baseline_correct"])/count,place_value=model.place.tolist())
    (here/"summary.json").write_text(json.dumps(summary,indent=2))
    print(summary)

if __name__=="__main__":
    main()
