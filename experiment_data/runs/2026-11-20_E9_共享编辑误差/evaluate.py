"""Score all original shared edits and preserve every CPU / GPU node value."""
from extract import *
from iterate import truth

def calibration(z):
    x=z-z.mean(0)
    w=np.linalg.lstsq(x,np.arange(10)-4.5,rcond=1e-7)[0]
    return z.mean(0),w,4.5

def main():
    model=Candidate(42,True,True)
    rows=formal_rows()
    base=(model.mean,model.weights,model.bias)
    output=[]
    for path in sorted(OUT.glob("*_*.npz")):
        tag=path.stem
        data=np.load(path)
        cals={"archived":base}
        if "canonical" in data and np.isfinite(data["canonical"]).all():
            cals["matched"]=calibration(data["canonical"])
        for label,(mean,w,bias) in cals.items():
            for prev in shared():
                line=prev["source_line"]
                r=rows[line-1]
                gold=build(r["after"],r["graph"],truth)
                k=r["target"]; ns=["s"]+[f"x{i}" for i in range(r["width"])]
                before=build([model.encode(n,v) for n,v in zip(ns,r["inputs"])],r["graph"],model.execute)
                digits=(data[str(line)]-mean)@w+bias
                value=number(model,digits)
                local,dirty=edit(model,before,k,value,r["graph"],len(ns))
                rounded=[round(v) if np.isfinite(v) else None for v in local]
                diffs=[i for i,(p,q) in enumerate(zip(rounded,gold)) if p!=q]
                def safe(v):
                    return float(v) if np.isfinite(v) else None
                output.append(dict(source_line=line,condition=tag,calibration=label,edited=r["after"][k],graph=r["graph"],
                    width=r["width"],mixed=r["mixed"],truth=gold[-1],pred=rounded[-1],correct=rounded[-1]==gold[-1],
                    decoded_digits=[safe(d) for d in digits],decoded_edit=safe(value),exact_nodes=gold,
                    predicted_nodes=[safe(v) for v in local],rounded_nodes=rounded,
                    affected=sorted(dirty),first_wrong_node=diffs[0] if diffs else None,
                    archived_gpu_pred=prev["pred"]))
    def stats(key):
        rs=[r for r in output if r["condition"]+"/"+r["calibration"]==key]
        return dict(n=len(rs),correct=sum(r["correct"] for r in rs),nonfinite=sum(r["pred"] is None for r in rs),
                    errors=[r["source_line"] for r in rs if not r["correct"]])
    keys=sorted({r["condition"]+"/"+r["calibration"] for r in output})
    summary={k:stats(k) for k in keys}
    linked=[]
    for prev in shared():
        if prev["truth"]==prev["pred"]:continue
        line=prev["source_line"]
        subset=[r for r in output if r["source_line"]==line and r["calibration"]=="archived"]
        gpu=next(r for r in subset if r["condition"]=="cuda_fp32_dynamic")
        cpu=next(r for r in subset if r["condition"]=="cpu_fp32_dynamic")
        idx=gpu["first_wrong_node"]
        linked.append(dict(source_line=line,edited=gpu["edited"],truth=gpu["truth"],original_gpu_pred=prev["pred"],
                           gpu_root=gpu["predicted_nodes"][-1],cpu_root=cpu["predicted_nodes"][-1],
                           gpu_input=gpu["decoded_edit"],cpu_input=cpu["decoded_edit"],
                           first_wrong_node=idx,first_wrong_exact=gpu["exact_nodes"][idx] if idx is not None else None,
                           first_wrong_value=gpu["predicted_nodes"][idx] if idx is not None else None))
    (OUT/"config.json").write_text(json.dumps(dict(source="P2-3 cuda.jsonl shared subset, same 36 source lines, widths, starting workspace and integer targets",fit_seed=42,
        device="NVIDIA GeForce RTX 4060 Laptop GPU; CPU fp32 four threads",precision="Original path uses bf16 input embeddings + FP32 blocks, LoRA and projection. Reduced conditions use torch autocast CUDA fp16/bf16 around that same encoder. Scalar fitting and graph propagation remain float64.",
        dynamic="Encode digits of the edited literal together at variable batch size 1..3 and native token length.",
        fixed="96-token right padding, fill batch to eight with s = 0, preserving actual masks. Archive keeps old readout. Matched recalibrates using ten labeled canonical digit representations from fixed 8x96, and keeps place/operation coefficients and initial workspace frozen.",source_line="1-based formal_rows_42.jsonl; 0-based node indexes; exact_nodes ordered as graph schedule."),indent=2))
    (OUT/"rows.jsonl").write_text("".join(json.dumps(r,allow_nan=False)+"\n" for r in output))
    (OUT/"summary.json").write_text(json.dumps(dict(by_condition=summary,original_two=linked),indent=2))
    print([(k,v["correct"]) for k,v in summary.items()]);print(linked)
if __name__=="__main__":main()
