"""Development and locked evaluation of a typed numerical channel on edit DAGs."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import random
import numpy as np

out=Path(__file__).resolve().parent
# Only the generator/reference use these functions. Candidate programs receive op IDs.
def truth(op,l,r):
    return (l+r if op==0 else l-r if op==1 else l*r)


def topology(width,mixed,rng):
    level=[]; nodes=[]
    for i in range(width):
        level.append(width+1+len(nodes))
        nodes.append((rng.randrange(3) if mixed else 0,0,i+1))
    rng.shuffle(level)
    while len(level)>1:
        following=[]
        for j in range(0,len(level)-1,2):
            following.append(width+1+len(nodes))
            nodes.append((rng.randrange(2) if mixed else 0,level[j],level[j+1]))
        if len(level)%2: following.append(level[-1])
        level=following
    return nodes


def build(values,nodes,executor):
    z=list(values)
    for op,l,r in nodes: z.append(executor(op,z[l],z[r]))
    return z


def fit(seed, bilinear):
    rng=random.Random(seed)
    pairs=[[] for _ in range(3)]
    for i in range(300):
        w=(2,4,8)[i%3]; vals=[rng.randrange(10) for _ in range(w+1)]
        nodes=topology(w,True,rng);z=build(vals,nodes,truth)
        for k,(op,l,r) in enumerate(nodes):
            pairs[op].append((z[l],z[r],z[w+1+k]))
    coefs=[]
    for grp in pairs:
        l,r,y=np.array(grp,dtype=float).T
        mat=np.stack([np.ones_like(l),l,r,l*r],1)
        if not bilinear: mat[:,3]=0
        coefs.append(np.linalg.lstsq(mat,y,rcond=None)[0])
    return np.array(coefs)


class Candidate:
    def __init__(self,seed,canonical,bilinear):
        raw=np.load(out/"inputs.npz")
        self.names=list(raw["names"])
        # Cache the local encoder output of each literal string. Numeric labels are only used at fit time.
        self.features={f"{name} = {j}":v.astype(float) for name, group in zip(self.names,raw["z"])
                       for j,v in enumerate(group)}
        self.canonical=canonical
        self.operators=fit(seed,bilinear)
        rng=random.Random(seed)
        examples=set()
        for _ in range(300):
            for n in self.names[:9]: examples.add((n,rng.randrange(10)))
        texts=[(f"{'s' if canonical else n} = {v}",v) for n,v in sorted(examples)]
        x=np.stack([self.features[t] for t,_ in texts])
        y=np.array([v for _,v in texts])
        self.mean=x.mean(0);self.bias=y.mean()
        self.weights=np.linalg.lstsq(x-self.mean,y-self.bias,rcond=1e-7)[0]
        self.decoded={}
        # Fit a shared recurrence on short labeled literals; extend it character by character.
        pairs=[(self.digit("s",s[0]),self.digit("s",s[1]),int(s)) for s in map(str,range(10,40))]
        l,r,y=np.array(pairs).T
        self.place=np.linalg.lstsq(np.stack([np.ones_like(l),l,r],1),y,rcond=None)[0]

    def digit(self,name,value):
        text=f"{name} = {value}"
        if self.canonical:
            text="s = "+text.split(" = ",1)[1]
        if text not in self.decoded:
            self.decoded[text]=float((self.features[text]-self.mean)@self.weights+self.bias)
        return self.decoded[text]

    def encode(self,name,value):
        chars=str(value)
        result=self.digit(name,chars[0])
        a,b,c=self.place
        for char in chars[1:]:
            result=a+b*result+c*self.digit(name,char)
        return result

    def execute(self,op,l,r):
        a,b,c,d=self.operators[op]
        return a+b*l+c*r+d*l*r


def evaluate(model, widths, count, seed, bounds=(0,9)):
    records=[]
    for mixed in (False,True):
        for w in widths:
            rng=random.Random(seed+int(mixed)*100+w)
            for i in range(count):
                nodes=topology(w,mixed,rng)
                low,high=bounds
                vals=[rng.randint(low,high) for _ in range(w+1)]
                names=["s"]+[f"x{n}" for n in range(w)]
                z=build([model.encode(n,v) for n,v in zip(names,vals)],nodes,model.execute)
                for target in (0,rng.randrange(1,w+1)):
                    new=vals.copy();new[target]=low+(new[target]-low+rng.randint(1,high-low))%(high-low+1)
                    gold=build(new,nodes,truth)[-1]
                    local=z.copy()
                    local[target]=model.encode(names[target],new[target])
                    dirty={target}
                    for j,(op,l,r) in enumerate(nodes,w+1):
                        if l in dirty or r in dirty:
                            local[j]=model.execute(op,local[l],local[r]);dirty.add(j)
                    rebuilt=build([model.encode(n,v) for n,v in zip(names,new)],nodes,model.execute)
                    pred=round(local[-1])
                    records.append(dict(mixed=mixed,width=w,task=i,target=target,truth=gold,
                                        pred=pred,rebuild=round(rebuilt[-1]),correct=pred==gold,
                                        actual=len(dirty)-1,total=len(nodes),
                                        preserved=all(z[j]==local[j] for j in range(len(z)) if j not in dirty),
                                        inputs=vals,after=new,graph=nodes))
    return records


def group(rows):
    keys=sorted({(r["mixed"],r["width"],r["target"]==0) for r in rows})
    result={}
    for m,w,shared in keys:
        rs=[r for r in rows if (r["mixed"],r["width"],r["target"]==0)==(m,w,shared)]
        result[f"{'mixed' if m else 'add'}/{w}/{'shared' if shared else 'private'}"]=dict(
            n=len(rs),correct=sum(r["correct"] for r in rs),agree=sum(r["pred"]==r["rebuild"] for r in rs),
            preserved=sum(r["preserved"] for r in rs))
    return result


def main():
    p=argparse.ArgumentParser();p.add_argument("phase",choices=["dev","formal"])
    phase=p.parse_args().phase
    lock=out/"frozen.json"
    if phase=="dev":
        results={}
        for canonical,bilinear in [(False,False),(True,False),(False,True),(True,True)]:
            label=f"canonical={canonical},bilinear={bilinear}"
            model=Candidate(42,canonical,bilinear)
            rs=evaluate(model,(2,4,8,10),120,731)
            stress=evaluate(model,(3,9),100,732,(40,59))
            stats=group(rs)|{"two_digit/"+k:v for k,v in group(stress).items()}
            results[label]=stats
            worst=min(g["correct"]/g["n"] for g in stats.values())
            print(label,"worst cell",worst,flush=True)
        (out/"development.json").write_text(json.dumps(results,indent=2))
        return
    config=json.loads(lock.read_text())
    assert config["script_sha256"]==hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    assert config["features_sha256"]==hashlib.sha256((out/"inputs.npz").read_bytes()).hexdigest()
    results={}
    for seed in config["fit_seeds"]:
        model=Candidate(seed,config["canonical"],config["bilinear"])
        rs=evaluate(model,config["widths"],config["tasks_per_cell"],config["evaluation_seed"])
        stats=group(rs)
        for label,bounds in config["extra_ranges"].items():
            extra=evaluate(model,config["widths"],config["tasks_per_cell"],config["evaluation_seed"]+bounds[0],bounds)
            stats.update({label+"/"+k:v for k,v in group(extra).items()})
            for r in extra:r["range"]=label
            rs.extend(extra)
        (out/f"formal_rows_{seed}.jsonl").write_text("".join(json.dumps(r)+"\n" for r in rs))
        results[seed]=dict(cells=stats,op_coefficients=model.operators.tolist(),place=model.place.tolist())
    (out/"formal.json").write_text(json.dumps(results,indent=2))


if __name__=="__main__":main()
