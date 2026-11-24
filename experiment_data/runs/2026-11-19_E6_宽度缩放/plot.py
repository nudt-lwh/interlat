from pathlib import Path
import os
folder=Path(__file__).resolve().parent

import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
s=json.loads((folder/"summary.json").read_text())["by_width"]
x=np.array(sorted(map(int,s)))
y=np.array([s[str(w)]["total_speedup"] for w in x])
interval=np.array([s[str(w)]["bootstrap_ci"] for w in x])
plt.rcParams.update({"font.family":"Arial","font.size":8,"pdf.fonttype":42,"svg.fonttype":"none","axes.spines.top":False,"axes.spines.right":False})
fig,ax=plt.subplots(figsize=(3.55,2.45),layout="constrained")
ax.axhline(1,ls="--",color="0.65",lw=.8)
ax.errorbar(x,y,yerr=np.vstack([y-interval[:,0],interval[:,1]-y]),color="#176D87",fmt="o-",linewidth=1.3,markersize=5,capsize=3)
ax.set(xticks=x, xlabel="Number of branches", ylabel="Private-edit speedup (×)",ylim=(0,max(y)*1.18),xlim=(14,106))
ax.tick_params(direction="out",length=3)
fig.savefig(folder/"width_scaling.pdf")
fig.savefig(folder/"width_scaling.svg")
fig.savefig(folder/"width_scaling.png", dpi=600)
