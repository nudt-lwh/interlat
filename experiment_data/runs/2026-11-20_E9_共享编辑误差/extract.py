"""Extract decoder inputs in the P2-3 device protocol and the E3 fixed-padding protocol."""
import json
from contextlib import nullcontext
from pathlib import Path
import sys
OUT=Path(__file__).resolve().parent
E7=OUT.parent/"2026-11-20_E7_算术批次不变性"
P23=OUT.parent/"2026-11-19_P2-3"
sys.path.insert(0,str(OUT.parent/"2026-11-19_P0-1"))
from encoder import FrozenEncoder,torch,np,formal_rows,number,edit,build,Candidate,method

def load_rows(path):
    return [json.loads(s) for s in path.read_text(encoding="utf-8").splitlines()]

def shared():
    return [r for r in load_rows(P23/"cuda.jsonl") if r["shared"]]

class Padding:
    def __init__(self,tok):
        self.tok=tok
        self.fixed=False
    def __call__(self,texts,**options):
        if self.fixed:
            options.update(padding="max_length",max_length=96,truncation=True)
        return self.tok(texts,**options)

def main(device):
    torch.set_num_threads(4)
    enc=FrozenEncoder(device)
    enc.tok=Padding(enc.tok)
    def forward(texts,fixed=False,precision="fp32"):
        enc.tok.fixed=fixed
        originals=len(texts)
        if fixed:
            texts=list(texts)+["s = 0"]*(8-len(texts))
        ctx=nullcontext() if precision=="fp32" else torch.autocast(device,dtype={"bf16":torch.bfloat16,"fp16":torch.float16}[precision])
        with ctx:
            # FrozenEncoder transfers to NumPy. bfloat16 needs a lossless conversion to float32 first.
            if precision=="bf16":
                hook=enc.enc.register_forward_hook(lambda m,x,y:y.float())
                try: z=enc(texts)
                finally: hook.remove()
            else: z=enc(texts)
        return z[:originals]
    canon=[f"s = {d}" for d in range(10)]
    if device=="cuda":
        items=[f"{n} = {v}" for n in ["s"]+[f"x{i}" for i in range(16)] for v in range(10)]
        maps={}
        for fixed in (False,True):
            for layout in ("archive","remixed"):
                # All 170 project literals. Remix adds longer distractors, changes order and native batch size.
                if layout=="archive": order=items.copy()
                else:
                    import random
                    order=items.copy()
                    random.Random(194720).shuffle(order)
                limit=8 if fixed or layout=="remixed" else 12
                stride=limit if layout=="archive" else limit-2
                by_text={}
                for start in range(0,len(order),stride):
                    group=order[start:start+stride]
                    batch=group if layout=="archive" else ["the value of a long unseen identifier is seven"]+group+["x15 = 9"]
                    vecs=forward(batch,fixed)
                    shift=int(layout=="remixed")
                    by_text.update(zip(group,vecs[shift:shift+len(group)]))
                key=("fixed" if fixed else "dynamic")+"_"+layout
                maps[key]=np.stack([by_text[t] for t in items])
            print("E7",fixed,flush=True)
        np.savez_compressed(E7/"features.npz",texts=items,**maps)
    for precision in (("fp32","fp16","bf16") if device=="cuda" else ("fp32",)):
        for fixed in ((False,True) if device=="cuda" else (False,)):
            tag=f"{device}_{precision}_{'fixed' if fixed else 'dynamic'}"
            vectors={}
            for row in shared():
                texts=[f"s = {d}" for d in row["text"].split(" = ")[1]]
                vectors[str(row["source_line"])]=forward(texts,fixed,precision)
            if fixed:
                vectors["canonical"]=np.concatenate([forward(canon[i:i+8],True,precision) for i in range(0,10,8)])
            np.savez_compressed(OUT/(tag+".npz"),**vectors)
            print("E9",tag,flush=True)
if __name__=="__main__":main(sys.argv[1])
