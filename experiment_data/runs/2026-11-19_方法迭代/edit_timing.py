"""Paired warm-state CPU edit latency on the locked arithmetic tasks."""
import csv
import json
import platform
import random
from collections import defaultdict
from pathlib import Path
from statistics import median
from time import perf_counter_ns

from iterate import Candidate, build

folder = Path(__file__).resolve().parent


def update(model, state, names, graph, target, new_value, local):
    values = state.copy()
    values[target] = model.encode(names[target], new_value)
    if local:
        dirty = {target}
        for node, (op, left, right) in enumerate(graph, len(names)):
            if left in dirty or right in dirty:
                values[node] = model.execute(op, values[left], values[right])
                dirty.add(node)
    else:
        for node, (op, left, right) in enumerate(graph, len(names)):
            values[node] = model.execute(op, values[left], values[right])
    return round(values[-1])


def main():
    model = Candidate(42, True, True)
    rows = [json.loads(line) for line in (folder / "formal_rows_42.jsonl").read_text().splitlines()]
    random.Random(1953).shuffle(rows)
    groups = defaultdict(list)
    with (folder / "edit_timing_rows.csv").open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(("width", "mixed", "task", "target", "inputs", "local_ns", "rebuild_ns"))
        for index, row in enumerate(rows):
            names = ["s"] + [f"x{k}" for k in range(row["width"])]
            inputs = [model.encode(n, v) for n, v in zip(names, row["inputs"])]
            graph = row["graph"]
            state = build(inputs, graph, model.execute)
            target = row["target"]
            args = (model, state, names, graph, target, row["after"][target])
            for local in (False, True):
                assert update(*args, local) == row["truth"]
            samples = {True: [], False: []}
            for repeat in range(6):
                first = bool((index + repeat) % 2)
                for local in (first, not first):
                    start = perf_counter_ns()
                    result = update(*args, local)
                    duration = perf_counter_ns() - start
                    assert result == row["truth"]
                    samples[local].append(duration)
            times = [median(samples[local]) for local in (True, False)]
            writer.writerow([row["width"], row["mixed"], row["task"], target, row["inputs"], *times])
            groups[(row["width"], target == 0)].append(times)
    report = []
    for (width, shared), times in sorted(groups.items()):
        local, rebuild = zip(*times)
        report.append(dict(width=width, shared=shared, edits=len(times),
                           median_local_us=median(local)/1000,
                           median_rebuild_us=median(rebuild)/1000,
                           paired_total_speedup=sum(rebuild)/sum(local)))
    result = dict(processor=platform.processor(), python=platform.python_version(),
                  template="six alternated repeats per edit; median per path per edit",
                  groups=report)
    (folder / "edit_timing.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
