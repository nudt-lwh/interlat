"""Fixed fork/join addition graphs, group-disjoint splits and reference execution."""
import itertools
import random


def schedule(width):
    inputs = ["s"] + [f"x{i}" for i in range(width)]
    level = [f"b{i}" for i in range(width)]
    ops = [(b, "add", ("s", x)) for b, x in zip(level, inputs[1:])]
    while len(level) > 1:
        following = []
        for lhs, rhs in zip(level[::2], level[1::2]):
            name = "g" if len(level) == 2 else f"m{len(ops)}"
            ops.append((name, "add", (lhs, rhs)))
            following.append(name)
        level = following
    return inputs, ops


def reference(width, values):
    inputs, ops = schedule(width)
    states = dict(zip(inputs, values))
    for name, _, (left, right) in ops:
        states[name] = states[left] + states[right]
    return states


def splits(width):
    rng = random.Random(101 + width)
    pool = []
    seen = set()
    size = min(4500, 10 ** (width + 1))
    while len(pool) < size:
        v = tuple(rng.randrange(10) for _ in range(width+1))
        if v not in seen and reference(width, v)["g"] <= 99:
            seen.add(v)
            pool.append(v)
    test, dev, train = pool[:200], pool[200:500], pool[500:]
    return train + rng.choices(train, k=4000-len(train)), dev, test


def changed(v, pos, rng):
    options = []
    for x in range(10):
        vv = (*v[:pos], x, *v[pos+1:])
        if x != v[pos] and reference(len(v)-1, vv)["g"] <= 99:
            options.append(vv)
    return rng.choice(options)


if __name__ == "__main__":
    for width in (2, 4, 8):
        train, dev, test = splits(width)
        assert not (set(train) & set(dev) or set(train) & set(test) or set(dev) & set(test))
        for values in test:
            for pos in (0, 1):
                vv = changed(values, pos, random.Random(7))
                expected = reference(width, values)["g"] + (vv[pos]-values[pos])*(width if pos==0 else 1)
                assert reference(width, vv)["g"] == expected
        print(width, len(set(train)), len(dev), len(test))
