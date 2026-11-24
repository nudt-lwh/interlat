import json
import os
from pathlib import Path

root = Path(__file__).resolve().parent
os.environ["MPLCONFIGDIR"] = str(root / "plot_cache")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

def save(fig,name):
    fig.savefig(root/(name+".png"), dpi=240, bbox_inches="tight")
    plt.close(fig)


def fork_chart():
    data=json.loads((root/"fork_summary.json").read_text())
    fig,axes=plt.subplots(1,3,figsize=(10,3.2),layout="constrained")
    for private, label, color in [(True,"Private input","#12668b"),(False,"Shared input","#a04727")]:
        rows=sorted([r for r in data if r["private"]==private],key=lambda r:r["width"])
        x=[r["width"] for r in rows]
        for ax, ys in zip(axes,([r["executed"] for r in rows],
                               [100*r["local"]/r["n"] for r in rows],
                               [r["speed"] for r in rows])):
            ax.plot(x,ys,"o-",label=label,color=color)
            ax.set_xticks(x);ax.set_xlabel("Number of branches")
        axes[2].errorbar(x,[r["speed"] for r in rows],
                         yerr=[[max(0,r["speed"]-r["speed_ci"][0]) for r in rows],
                               [max(0,r["speed_ci"][1]-r["speed"]) for r in rows]],
                         fmt="none",color=color,capsize=3)
    for ax,title in zip(axes,["Derived nodes executed","Exact answer accuracy (%)","End-to-end speedup"]):
        ax.set_title(title,fontsize=11)
        ax.spines[["right","top"]].set_visible(False)
    axes[1].set_ylim(0,102);axes[2].axhline(1,color="#777777",ls=":",lw=1)
    axes[0].legend(frameon=False)
    save(fig,"fork_results")


def belief_chart():
    data=[json.loads(line) for line in (root/"belief_rows.jsonl").read_text().splitlines()]
    fig,axes=plt.subplots(1,2,figsize=(6.8,3.2),layout="constrained")
    for ax,change in zip(axes,(True,False)):
        selected=[r for r in data if (r["gt0"]!=r["gt1"])==change]
        m=np.zeros((3,3),dtype=int)
        for r in selected:m[r["initial"],r["edited"]]+=1
        ax.imshow(m/max(1,len(selected)),cmap="Blues",vmin=0,vmax=1)
        for i in range(3):
            for j in range(3):
                ax.text(j,i,str(m[i,j]),ha="center",va="center",
                        color="white" if m[i,j]>.5*len(selected) else "#20252b")
        ax.set_title(("Update" if change else "Maintain")+f" (n={len(selected)})")
        ax.set_xticks(range(3),labels=list("abc"));ax.set_yticks(range(3),labels=list("abc"))
        ax.set_xlabel("Prediction after evidence");ax.set_ylabel("Initial prediction")
    save(fig,"belief_transitions")


if __name__=="__main__":
    fork_chart()
    belief_chart()
