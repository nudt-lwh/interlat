"""Render E2 from archived summary counts."""
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

out=Path(__file__).resolve().parent
figdir=out.parents[2]/"figures/exports"
figdir.mkdir(parents=True,exist_ok=True)
data=json.loads((out/"summary.json").read_text())["by_group_budget"]
plt.rcParams.update({"font.family":"DejaVu Sans","font.size":10.5,"svg.fonttype":"none","pdf.fonttype":42,
                     "axes.spines.top":False,"axes.spines.right":False,"legend.frameon":False})
fig,ax=plt.subplots(figsize=(6.2,4.2))
x=np.array([10,30,100])
ax.set_axisbelow(True)
ax.grid(axis="y",color="#d8dfe2",lw=0.6)
controls=[100*data[f"names/{n}"]["adapter_accuracy"] for n in x]
assert all(100*data[f"{g}/{n}"]["adapter_accuracy"]==v for n,v in zip(x,controls) for g in ("sentence","words"))
adapter,=ax.plot(x,controls,ls=(0,(4,2)),color="#546473",lw=1.8,label="Format adapter (all groups)")
lines=[]
for group,label,color,mark in [("names","Held-out name","#275e88","o"),("sentence","New sentence form","#c05a2b","s"),
                              ("words","English digit word","#008170","D")]:
    y=[100*data[f"{group}/{n}"]["accuracy"] for n in x]
    line,=ax.plot(x,y,marker=mark,color=color,lw=1.9,ms=6,label=label)
    lines.append(line)
ax.set(xscale="log",xlabel="Labeled examples per digit",ylabel="Single-digit exact match (%)",ylim=(-3,104),xlim=(8,125))
ax.set_xticks(x,labels=list(map(str,x)))
ax.minorticks_off()
ax.set_yticks(range(0,101,20))
fig.legend(handles=[*lines,adapter],loc="lower center",ncols=2,columnspacing=1.2,
           handlelength=2.6,fontsize=10,bbox_to_anchor=(0.52,0.01))
fig.subplots_adjust(left=.13,right=.98,top=.96,bottom=.31)
fig.savefig(figdir/"fig07_surface_learning.pdf",metadata={"CreationDate":None})
for ext in ["png","svg"]:
    fig.savefig(figdir/f"fig07_surface_learning.{ext}",dpi=600)
