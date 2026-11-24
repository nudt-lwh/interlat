from pathlib import Path
import sys
folder = Path(__file__).resolve().parent
sys.dont_write_bytecode = True
sys.path.insert(0, str(folder.parent / "2026-11-19_P0-2"))
import json
from collections import Counter
with (folder/"rows.jsonl").open() as f, (folder.parent/"2026-11-19_P2-2"/"rows.jsonl").open() as prior:
    rows=[json.loads(line) for line in f]
    source=[json.loads(line) for line in prior]
summary=json.loads((folder/"summary.json").read_text())
config=json.loads((folder/"config.json").read_text())
assert len(rows)==len(source)==10800
assert config["selected_alpha"]==float(max(sorted(config["dev_correct"], key=float,reverse=True),key=lambda k:(config["dev_correct"][k],-float(k))))
for label,s in summary["methods"].items():
    tests=[r[label] for r in rows]
    assert s["n"]==len(tests) and s["root_correct"]==sum(z["pred"]==z["gold"] for z in tests)
    assert s["correct_nodes"]==sum(z["correct_nodes"] for z in tests)
    assert s["nodes"]==sum(z["nodes"] for z in tests)
    assert s["first_wrong"]==Counter(z["first"]["op"] for z in tests if z["first"])
    assert s["first_wrong_on_root_failure"]==Counter(z["first"]["op"] for z in tests if z["first"] and z["gold"]!=z["pred"])
    assert all(r["source_line"]==i and r["symbolic"]==old["truth"]==r[label]["gold"] and old["pred"]==r["shared_ols_pred"] for i,(r,old) in enumerate(zip(rows,source),1))
print("Verified paired predictions, reference answers, node denominators, first errors, and development-only selection.")